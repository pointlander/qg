// Copyright 2026 The QG Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
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
	Width = 4
)

// Hadamard computes the hadamard product of two tensors
func Hadamard(k tc128.Continuation, node int, a, b *tc128.V, options ...map[string]interface{}) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	length := len(b.X)
	c := tc128.NewV(a.S...)
	for i, j := range a.X {
		c.X = append(c.X, j*b.X[i%length])
	}
	if k(&c) {
		return true
	}
	for i, j := range c.D {
		a.D[i] += j * b.X[i%length]
		b.D[i%length] += j * a.X[i]
	}
	return false
}

// G is a g model
type G struct {
	Iteration int
	Rng       *rand.Rand
	G         Matrix[complex128]
	Others    *tc128.Set
	Set       *tc128.Set
}

// NewG creates a new g model
func NewG(rows, cols int) G {
	rng := rand.New(rand.NewSource(1))

	g := NewMatrix[complex128](Width-1, 33)
	for range g.Rows {
		for range g.Cols {
			g.Data = append(g.Data, complex(rng.Float64(), rng.Float64()))
		}
	}

	others := tc128.NewSet()
	others.Add("x", cols, rows)
	x := others.ByName["x"]
	x.X = x.X[:cap(x.X)]

	set := tc128.NewSet()
	set.Add("i", Width, rows)
	set.Add("g", cols, rows)
	//set.Add("l", 1, 1)

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
	for i := range set.ByName["g"].X {
		set.ByName["g"].X[i] = complex(1e-5, 1e-5)
	}
	//set.ByName["l"].X[0] = U
	return G{
		Rng:    rng,
		G:      g,
		Others: &others,
		Set:    &set,
	}
}

// Iterate iterates the g model
func (g *G) Iterate(iterations int) ([]complex128, Matrix[complex128]) {
	inputs := NewMatrix[complex128](g.G.Rows, g.G.Rows)
	for i := range g.G.Rows {
		for j := range g.G.Rows {
			sum := 0.0
			for k := range g.G.Cols {
				diff := cmplx.Abs(g.G.Data[i*g.G.Cols+k] - g.G.Data[j*g.G.Cols+k])
				sum += diff * diff
			}
			distance := math.Sqrt(sum)
			if distance == 0 {
				inputs.Data = append(inputs.Data, 0)
				continue
			}
			inputs.Data = append(inputs.Data, complex(1/distance, 0))
		}
	}
	x, index := g.Others.ByName["x"], 0
	for row := range inputs.Rows {
		for _, value := range inputs.Data[row*inputs.Cols : row*inputs.Cols+inputs.Cols] {
			x.X[index] = value
			index++
		}
	}

	drop := .3
	dropout := map[string]interface{}{
		"rng":  g.Rng,
		"drop": &drop,
	}

	hadamard := tc128.B(Hadamard)
	//c := tf64.Inv(hadamard(set.Get("l"), set.Get("g")))
	//c := tf64.Inv(others.Get("c"))
	sa := tc128.Mul(tc128.Dropout(tc128.Square( /*hadamard(*/ g.Set.Get("i") /*, c)*/), dropout), hadamard(g.Others.Get("x"), g.Set.Get("g")))
	loss := tc128.Avg(tc128.Quadratic(tc128.Mul(hadamard(g.Others.Get("x"), g.Set.Get("g")), tc128.Dropout(tc128.Square( /*hadamard(*/ g.Set.Get("i") /*, c)*/), dropout)), sa))

	var l complex128
	iteration := g.Iteration
	pow := func(x float64) float64 {
		y := math.Pow(x, float64(iteration+1))
		if math.IsNaN(y) || math.IsInf(y, 0) {
			return 0
		}
		return y
	}

	g.Set.Zero()
	g.Others.Zero()
	l = tc128.Gradient(loss).X[0]
	if cmplx.IsNaN(l) || cmplx.IsInf(l) {
		fmt.Println(iteration, l)
		return nil, Matrix[complex128]{}
	}

	norm := 0.0
	for _, p := range g.Set.Weights {
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
	for _, w := range g.Set.Weights {
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
			_ = mhat
			w.X[ii] -= Eta * mhat / (cmplx.Sqrt(vhat) + 1e-8)
			/*if rng.Float64() > .01 {
				w.X[ii] -= .05 * g
			} else {
				w.X[ii] += .05 * g
			}*/
		}
	}
	g.Iteration++
	fmt.Println(l)

	/*meta := make([][]float64, len(cp))
	for i := range meta {
		meta[i] = make([]float64, len(cp))
	}
	const k = 3

	{
		y := set.ByName["i"]
		vectors := make([][]float64, len(cp))
		for i := range vectors {
			row := make([]float64, Width)
			for ii := range row {
				row[ii] = y.X[i*Width+ii]
			}
			vectors[i] = row
		}
		for i := 0; i < 33; i++ {
			clusters, _, err := kmeans.Kmeans(int64(i+1), vectors, k, kmeans.SquaredEuclideanDistance, -1)
			if err != nil {
				panic(err)
			}
			for i := 0; i < len(meta); i++ {
				target := clusters[i]
				for j, v := range clusters {
					if v == target {
						meta[i][j]++
					}
				}
			}
		}
	}
	clusters, _, err := kmeans.Kmeans(1, meta, 3, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i := range clusters {
		cp[i].Cluster = clusters[i]
	}
	for _, value := range x.X[len(iris)*size:] {
		cp[len(iris)].Measures = append(cp[len(iris)].Measures, value)
	}
	I := set.ByName["i"]
	for i := range cp {
		cp[i].Embedding = I.X[i*Width : (i+1)*Width]
	}
	sort.Slice(cp, func(i, j int) bool {
		return cp[i].Cluster < cp[j].Cluster
	})*/

	I := g.Set.ByName["i"]
	for i := range I.S[1] {
		v := I.X[i*I.S[0]+3]
		for ii := range I.S[0] - 1 {
			g.G.Data[i*g.G.Cols+ii] += v * I.X[i*I.S[0]+ii]
		}
	}

	return g.Set.ByName["g"].X, g.G
}

// SMode s mode
func SMode(epochs int, iterate func(iterations int) ([]complex128, Matrix[complex128])) {
	images := &gif.GIF{}
	var palette = []color.Color{}
	for i := range 256 {
		g := byte(i)
		palette = append(palette, color.RGBA{g, g, g, 0xff})
	}
	/*delay := make([][]chan float64, g.Rows)
	for i := range delay {
		delay[i] = make([]chan float64, g.Cols)
		for ii := range delay[i] {
			delay[i][ii] = make(chan float64, 2)
		}
	}*/
	gs := make(plotter.XYs, 0, 8)
	gavg := make(plotter.XYs, 0, 8)
	var gshist plotter.Values
	for epoch := range epochs {
		fmt.Println(epoch)
		G, g := iterate(512)
		if epoch < 1024 {
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
				for i := range g.Rows {
					x, y := cmplx.Abs(g.Data[i*g.Cols+offset.A]), cmplx.Abs(g.Data[i*g.Cols+offset.B])
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
				for i := range g.Rows {
					xx, yy := cmplx.Abs(g.Data[i*g.Cols+offset.A]), cmplx.Abs(g.Data[i*g.Cols+offset.B])
					x := 500*(xx-minX)/(maxX-minX) + 6
					y := 500*(yy-minY)/(maxY-minY) + 6
					image.Set(offset.X+int(x), offset.Y+int(y), color.RGBA{0xff, 0xff, 0xff, 0xff})
				}
			}
			images.Image = append(images.Image, image)
			images.Delay = append(images.Delay, 10)
		}
		avg := complex128(0.0)
		for _, value := range G {
			avg += value
		}
		avg /= complex(float64(len(G)), 0)
		stddev := complex128(0.0)
		for _, value := range G {
			diff := value - avg
			stddev = diff * diff
		}
		stddev /= complex(float64(len(G)), 0)
		stddev = cmplx.Sqrt(stddev)
		fmt.Println("G", avg, stddev)
		gavg = append(gavg, plotter.XY{X: real(avg), Y: imag(avg)})
		for _, G := range G {
			gs = append(gs, plotter.XY{X: float64(epoch), Y: float64(cmplx.Abs(G))})
			gshist = append(gshist, float64(cmplx.Abs(G)))
		}
		//gg = G
	}

	{
		out, err := os.Create("verse.gif")
		if err != nil {
			panic(err)
		}
		defer out.Close()
		err = gif.EncodeAll(out, images)
		if err != nil {
			panic(err)
		}
	}

	{
		p := plot.New()

		p.Title.Text = "G vs time"
		p.X.Label.Text = "time"
		p.Y.Label.Text = "G"

		scatter, err := plotter.NewScatter(gs)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		p.Add(scatter)

		err = p.Save(8*vg.Inch, 8*vg.Inch, "G.png")
		if err != nil {
			panic(err)
		}
	}

	{
		p := plot.New()

		p.Title.Text = "G vs time"
		p.X.Label.Text = "time"
		p.Y.Label.Text = "G"

		scatter, err := plotter.NewScatter(gavg)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		p.Add(scatter)

		err = p.Save(8*vg.Inch, 8*vg.Inch, "Gavg.png")
		if err != nil {
			panic(err)
		}
	}

	{
		/*p := plot.New()
		p.Title.Text = "G"

		hist, err := plotter.NewHist(gshist, 256)
		if err != nil {
			panic(err)
		}
		max, index := 0.0, 0
		for i, bin := range hist.Bins {
			if bin.Weight > max {
				max, index = bin.Weight, i
			}
		}
		{
			min, max := hist.Bins[index].Min, hist.Bins[index].Max
			fmt.Println("min max", min, max)
			fmt.Println("min^2 max^2", min*min, max*max)
			fmt.Println("min/c max/c", min/V, max/V)
		}
		p.Add(hist)

		err = p.Save(8*vg.Inch, 8*vg.Inch, "Ghist.png")
		if err != nil {
			panic(err)
		}

		sort.Slice(hist.Bins, func(i, j int) bool {
			return hist.Bins[i].Weight < hist.Bins[j].Weight
		})
		for i := range hist.Bins {
			fmt.Println(hist.Bins[i])
		}*/

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

func main() {
	g := NewG(33, 33)
	SMode(1024, g.Iterate)
}
