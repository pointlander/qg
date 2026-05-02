// Copyright 2026 The QG Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/gif"
	"math"
	"math/cmplx"
	"math/rand"
	"os"
	"sort"
	"strings"

	"github.com/pointlander/gradient/tc128"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-3
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

const (
	// Width is the embedding width
	Width = 3
)

var palette = []color.Color{}

func init() {
	for i := range 256 {
		g := byte(i)
		palette = append(palette, color.RGBA{g, g, g, 0xff})
	}
}

// Euclidean computes the euclidean distance between all row vectors and all row vectors
func Euclidean(k tc128.Continuation, node int, a, b *tc128.V, options ...map[string]interface{}) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width := a.S[0]
	if width != b.S[0] || a.S[1] != b.S[1] {
		panic("dimensions are not the same")
	}
	c, sizeA, sizeB := tc128.NewV(a.S[1], b.S[1]), len(a.X), len(b.X)
	for i := 0; i < sizeA; i += width {
		for ii := 0; ii < sizeB; ii += width {
			av, bv, sum := a.X[i:i+width], b.X[ii:ii+width], complex128(0.0)
			for j, ax := range av {
				diff := (ax - bv[j])
				sum += diff * diff
			}
			c.X = append(c.X, cmplx.Sqrt(sum))
		}
	}
	if k(&c) {
		return true
	}
	index := 0
	for i := 0; i < sizeA; i += width {
		for ii := 0; ii < sizeB; ii += width {
			av, bv, cx, ad, bd, d := a.X[i:i+width], b.X[ii:ii+width], c.X[index], a.D[i:i+width], b.D[ii:ii+width], c.D[index]
			for j, ax := range av {
				if cx == 0 {
					continue
				}
				ad[j] += (ax - bv[j]) * d / cx
				bd[j] += (bv[j] - ax) * d / cx
			}
			index++
		}
	}
	return false
}

// QG is a quantum gravity model
type QG struct {
	Iteration int
	Rng       *rand.Rand
	X         Matrix[complex128]
	Others    *tc128.Set
	Set       *tc128.Set
	Loss      plotter.XYs
	Images    *gif.GIF
}

// NewQG creates a new quantum gravity model
func NewQG(rows, cols int) QG {
	rng := rand.New(rand.NewSource(1))

	x := NewMatrix[complex128](Width, rows)
	for range x.Rows {
		for range x.Cols {
			x.Data = append(x.Data, complex(rng.Float64(), rng.Float64()))
		}
	}

	others := tc128.NewSet()
	others.Add("x", cols, rows)
	{
		x := others.ByName["x"]
		x.X = x.X[:cap(x.X)]
	}

	set := tc128.NewSet()
	set.Add("v", Width, rows)
	set.Add("g", cols, rows)

	for ii := range set.Weights {
		w := set.Weights[ii]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]complex128, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]complex128, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for range cap(w.X) {
			w.X = append(w.X, complex(rng.NormFloat64()*factor, rng.NormFloat64()*factor))
		}
		w.States = make([][]complex128, StateTotal)
		for ii := range w.States {
			w.States[ii] = make([]complex128, len(w.X))
		}
	}

	return QG{
		Rng:    rng,
		X:      x,
		Others: &others,
		Set:    &set,
		Loss:   make(plotter.XYs, 0, 8),
		Images: &gif.GIF{},
	}
}

// Iterate iterates the g model
func (q *QG) Iterate(iterations int) *tc128.V {
	x, index := q.Others.ByName["x"], 0
	for i := range q.X.Rows {
		for j := range q.X.Rows {
			sum := 0.0
			for k := range q.X.Cols {
				diff := cmplx.Abs(q.X.Data[i*q.X.Cols+k] - q.X.Data[j*q.X.Cols+k])
				sum += diff * diff
			}
			distance := math.Sqrt(sum)
			if distance == 0 {
				x.X[index] = 0
				index++
				continue
			}
			x.X[index] = complex(1/distance, 0)
			index++
		}
	}

	drop := .3
	dropout := map[string]interface{}{
		"rng":  q.Rng,
		"drop": &drop,
	}

	l0 := tc128.Mul(tc128.Dropout(tc128.Square(q.Set.Get("v")), dropout),
		tc128.Hadamard(q.Others.Get("x"), q.Set.Get("g")))
	loss := tc128.Avg(tc128.Quadratic(tc128.Mul(tc128.Hadamard(q.Others.Get("x"), q.Set.Get("g")),
		tc128.Dropout(tc128.Square(q.Set.Get("v")), dropout)), l0))

	var l complex128
	for range iterations {
		iteration := q.Iteration
		pow := func(x float64) float64 {
			y := math.Pow(x, float64(iteration+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return y
		}

		q.Set.Zero()
		q.Others.Zero()
		l = tc128.Gradient(loss).X[0]
		if cmplx.IsNaN(l) || cmplx.IsInf(l) {
			fmt.Println(iteration, l)
			return nil
		}

		norm := 0.0
		for _, p := range q.Set.Weights {
			for _, d := range p.D {
				norm += cmplx.Abs(d) * cmplx.Abs(d)
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1), pow(B2)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range q.Set.Weights {
			for ii, d := range w.D {
				g := d * complex(scaling, 0)
				m := B1*w.States[StateM][ii] + (1-B1)*g
				v := B2*w.States[StateV][ii] + (1-B2)*g*g
				w.States[StateM][ii] = m
				w.States[StateV][ii] = v
				mhat := m / (1 - complex(b1, 0))
				vhat := v / (1 - complex(b2, 0))
				if cmplx.Abs(vhat) < 0 {
					vhat = 0
				}
				w.X[ii] -= Eta * mhat / (cmplx.Sqrt(vhat) + 1e-8)
			}
		}
		q.Loss = append(q.Loss, plotter.XY{X: float64(iteration), Y: math.Log10(cmplx.Abs(l))})
		q.Iteration++
	}
	fmt.Println(l)

	v := q.Set.ByName["v"]
	for i := range v.S[1] {
		/*dt := v.X[i*v.S[0]+3]
		for ii := range v.S[0] - 1 {
			q.X.Data[i*q.X.Cols+ii] += dt * v.X[i*v.S[0]+ii]
		}*/
		q.X.Data[i*q.X.Cols+0] += v.X[i*v.S[0]+0] * v.X[i*v.S[0]+1]
		q.X.Data[i*q.X.Cols+1] += v.X[i*v.S[0]+0] * v.X[i*v.S[0]+2]
		q.X.Data[i*q.X.Cols+2] += v.X[i*v.S[0]+1] * v.X[i*v.S[0]+2]
	}
	if q.Iteration < 1024 {
		image := image.NewPaletted(image.Rect(0, 0, 1024, 1024), palette)
		type Offset struct {
			X int
			Y int
			A int
			B int
		}
		offsets := []Offset{{0, 0, 0, 1}, {512, 0, 0, 2}, {0, 512, 1, 2}}
		for _, offset := range offsets {
			minX, maxX, minY, maxY := math.MaxFloat64, -math.MaxFloat64, math.MaxFloat64, -math.MaxFloat64
			for i := range q.X.Rows {
				x, y := cmplx.Abs(q.X.Data[i*q.X.Cols+offset.A]), cmplx.Abs(q.X.Data[i*q.X.Cols+offset.B])
				if x < minX {
					minX = x
				}
				if x > maxX {
					maxX = x
				}
				if y < minY {
					minY = y
				}
				if y > maxY {
					maxY = y
				}
			}
			for i := range q.X.Rows {
				xx, yy := cmplx.Abs(q.X.Data[i*q.X.Cols+offset.A]), cmplx.Abs(q.X.Data[i*q.X.Cols+offset.B])
				x := 500*(xx-minX)/(maxX-minX) + 6
				y := 500*(yy-minY)/(maxY-minY) + 6
				image.Set(offset.X+int(x), offset.Y+int(y), color.RGBA{0xff, 0xff, 0xff, 0xff})
			}
			for i := range 1024 {
				image.Set(512, i, color.RGBA{0xff, 0xff, 0xff, 0xff})
				image.Set(i, 512, color.RGBA{0xff, 0xff, 0xff, 0xff})
			}
			for i := range 512 {
				for ii := range 4 {
					image.Set(int(float64(q.Iteration*i)/float64(1024))+512, 1023-ii, color.RGBA{0xff, 0xff, 0xff, 0xff})
				}
			}
		}
		q.Images.Image = append(q.Images.Image, image)
		q.Images.Delay = append(q.Images.Delay, 10)
	}

	return q.Set.ByName["g"]
}

// Q is a quantum gravity model
type Q struct {
	Iteration int
	Rng       *rand.Rand
	Set       *tc128.Set
	Loss      plotter.XYs
	Images    *gif.GIF
}

// NewQG creates a new quantum gravity model
func NewQ(rows, cols int) Q {
	rng := rand.New(rand.NewSource(1))

	set := tc128.NewSet()
	set.Add("v", 2, rows)
	set.Add("g", cols, rows)
	//set.Add("x", 2, rows)

	for ii := range set.Weights {
		w := set.Weights[ii]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]complex128, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]complex128, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0/float64(w.S[0])) * .1
		for range cap(w.X) {
			w.X = append(w.X, complex(rng.NormFloat64()*factor, rng.NormFloat64()*factor))
		}
		w.States = make([][]complex128, StateTotal)
		for ii := range w.States {
			w.States[ii] = make([]complex128, len(w.X))
		}
	}

	return Q{
		Rng:    rng,
		Set:    &set,
		Loss:   make(plotter.XYs, 0, 8),
		Images: &gif.GIF{},
	}
}

// Iterate iterates the g model
func (q *Q) Iterate(iterations int) *tc128.V {
	drop := .3
	dropout := map[string]interface{}{
		"rng":  q.Rng,
		"drop": &drop,
	}

	euclidean := tc128.B(Euclidean)

	l0 := tc128.Mul(tc128.Dropout(tc128.Square(q.Set.Get("v")), dropout),
		tc128.Hadamard(tc128.Inv(euclidean(q.Set.Get("v"), q.Set.Get("v"))), q.Set.Get("g")))
	loss := tc128.Avg(tc128.Quadratic(tc128.Mul(tc128.Hadamard(tc128.Inv(euclidean(q.Set.Get("v"), q.Set.Get("v"))), q.Set.Get("g")),
		tc128.Dropout(tc128.Square(q.Set.Get("v")), dropout)), l0))

	var l complex128
	for range iterations {
		iteration := q.Iteration
		pow := func(x float64) float64 {
			y := math.Pow(x, float64(iteration+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return y
		}

		q.Set.Zero()
		l = tc128.Gradient(loss).X[0]
		if cmplx.IsNaN(l) || cmplx.IsInf(l) {
			fmt.Println(iteration, l)
			return nil
		}

		norm := 0.0
		for _, p := range q.Set.Weights {
			for _, d := range p.D {
				norm += cmplx.Abs(d) * cmplx.Abs(d)
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1), pow(B2)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range q.Set.Weights {
			for ii, d := range w.D {
				g := d * complex(scaling, 0)
				m := B1*w.States[StateM][ii] + (1-B1)*g
				v := B2*w.States[StateV][ii] + (1-B2)*g*g
				w.States[StateM][ii] = m
				w.States[StateV][ii] = v
				mhat := m / (1 - complex(b1, 0))
				vhat := v / (1 - complex(b2, 0))
				if cmplx.Abs(vhat) < 0 {
					vhat = 0
				}
				w.X[ii] -= Eta * mhat / (cmplx.Sqrt(vhat) + 1e-8)
			}
		}
		q.Loss = append(q.Loss, plotter.XY{X: float64(iteration), Y: math.Log10(cmplx.Abs(l))})
		q.Iteration++
	}
	fmt.Println(l)

	v := q.Set.ByName["v"]
	if q.Iteration < 1024 {
		image := image.NewPaletted(image.Rect(0, 0, 512, 512), palette)
		type Offset struct {
			X int
			Y int
			A int
			B int
		}
		minX, maxX, minY, maxY := math.MaxFloat64, -math.MaxFloat64, math.MaxFloat64, -math.MaxFloat64
		for i := range v.S[1] {
			x, y := cmplx.Abs(v.X[i*v.S[0]]), cmplx.Abs(v.X[i*v.S[0]+1])
			if x < minX {
				minX = x
			}
			if x > maxX {
				maxX = x
			}
			if y < minY {
				minY = y
			}
			if y > maxY {
				maxY = y
			}
		}
		for i := range v.S[1] {
			xx, yy := cmplx.Abs(v.X[i*v.S[0]]), cmplx.Abs(v.X[i*v.S[0]+1])
			x := 500*(xx-minX)/(maxX-minX) + 6
			y := 500*(yy-minY)/(maxY-minY) + 6
			image.Set(int(x), int(y), color.RGBA{0xff, 0xff, 0xff, 0xff})
		}
		for i := range 1024 {
			image.Set(512, i, color.RGBA{0xff, 0xff, 0xff, 0xff})
			image.Set(i, 512, color.RGBA{0xff, 0xff, 0xff, 0xff})
		}
		for i := range 512 {
			for ii := range 4 {
				image.Set(int(float64(q.Iteration*i)/float64(1024)), 511-ii, color.RGBA{0xff, 0xff, 0xff, 0xff})
			}
		}
		q.Images.Image = append(q.Images.Image, image)
		q.Images.Delay = append(q.Images.Delay, 10)
	}

	return q.Set.ByName["g"]
}

// Simulate runs the simulation
func Simulate(prefix string, epochs int, iterate func(iterations int) *tc128.V) {
	gs := make(plotter.XYs, 0, 8)
	gavg := make(plotter.XYs, 0, 8)
	var gshist plotter.Values
	for epoch := range epochs {
		fmt.Println(epoch)
		g := iterate(1)
		avg := complex128(0.0)
		for _, value := range g.X {
			avg += value
		}
		avg /= complex(float64(len(g.X)), 0)
		stddev := complex128(0.0)
		for _, value := range g.X {
			diff := value - avg
			stddev = diff * diff
		}
		stddev /= complex(float64(len(g.X)), 0)
		stddev = cmplx.Sqrt(stddev)
		fmt.Println("G", avg, stddev)
		gavg = append(gavg, plotter.XY{X: real(avg), Y: imag(avg)})
		for _, v := range g.X {
			gs = append(gs, plotter.XY{X: real(v), Y: imag(v)})
			gshist = append(gshist, float64(cmplx.Abs(v)))
		}
	}

	{
		p := plot.New()

		p.Title.Text = "imag vs real"
		p.X.Label.Text = "real"
		p.Y.Label.Text = "imag"

		scatter, err := plotter.NewScatter(gs)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		p.Add(scatter)

		err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("%sG.png", prefix))
		if err != nil {
			panic(err)
		}
	}

	{
		p := plot.New()

		p.Title.Text = "imag vs real"
		p.X.Label.Text = "real"
		p.Y.Label.Text = "imag"

		scatter, err := plotter.NewScatter(gavg)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		p.Add(scatter)

		err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("%sGavg.png", prefix))
		if err != nil {
			panic(err)
		}
	}

	{
		fmt.Println()
		histogram := make(map[int]int)
		for _, value := range gshist {
			exp := int(math.Floor(math.Log10(math.Abs(value))))
			count := histogram[exp]
			count++
			histogram[exp] = count
		}
		type Count struct {
			Count int
			Exp   int
		}
		counts := make([]Count, 0, len(histogram))
		for key, value := range histogram {
			counts = append(counts, Count{
				Count: value,
				Exp:   key,
			})
		}
		sort.Slice(counts, func(i, j int) bool {
			return counts[i].Count < counts[j].Count
		})
		for _, count := range counts {
			fmt.Println(count.Exp, ":", count.Count)
		}
	}
}

var (
	// FlagEpochs number of epochs
	FlagEpochs = flag.Int("e", 1, "number of epochs")
	// FlagQ q mode
	FlagQ = flag.Bool("q", false, "q mode")
)

func main() {
	flag.Parse()

	if *FlagQ {
		q := NewQ(33, 33)
		prefix := "q_"
		Simulate(prefix, *FlagEpochs*1024, q.Iterate)

		{
			out, err := os.Create(fmt.Sprintf("%sverse.gif", prefix))
			if err != nil {
				panic(err)
			}
			defer out.Close()
			err = gif.EncodeAll(out, q.Images)
			if err != nil {
				panic(err)
			}
		}

		{
			p := plot.New()

			p.Title.Text = "loss vs iteration"
			p.X.Label.Text = "iteration"
			p.Y.Label.Text = "log loss"

			scatter, err := plotter.NewScatter(q.Loss)
			if err != nil {
				panic(err)
			}
			scatter.GlyphStyle.Radius = vg.Length(1)
			scatter.GlyphStyle.Shape = draw.CircleGlyph{}
			p.Add(scatter)

			err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("%sloss.png", prefix))
			if err != nil {
				panic(err)
			}
		}
		return
	}

	q := NewQG(33, 33)
	prefix := ""
	Simulate(prefix, *FlagEpochs*1024, q.Iterate)

	{
		out, err := os.Create(fmt.Sprintf("%sverse.gif", prefix))
		if err != nil {
			panic(err)
		}
		defer out.Close()
		err = gif.EncodeAll(out, q.Images)
		if err != nil {
			panic(err)
		}
	}

	{
		p := plot.New()

		p.Title.Text = "loss vs iteration"
		p.X.Label.Text = "iteration"
		p.Y.Label.Text = "log loss"

		scatter, err := plotter.NewScatter(q.Loss)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		p.Add(scatter)

		err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("%sloss.png", prefix))
		if err != nil {
			panic(err)
		}
	}
}
