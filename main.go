// Copyright 2026 The QG Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"strings"

	"github.com/pointlander/gradient/tc128"
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
	Others    *tc128.Set
	Set       *tc128.Set
}

// NewG creates a new g model
func NewG(rows, cols int) G {
	rng := rand.New(rand.NewSource(1))

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
			w.X = append(w.X, complex(rng.NormFloat64()*factor, 0))
		}
		w.States = make([][]complex128, StateTotal)
		for ii := range w.States {
			w.States[ii] = make([]complex128, len(w.X))
		}
	}
	for i := range set.ByName["g"].X {
		set.ByName["g"].X[i] = 1e-5
	}
	//set.ByName["l"].X[0] = U
	return G{
		Rng:    rng,
		Others: &others,
		Set:    &set,
	}
}

// Iterate iterates the g model
func (g *G) Iterate(inputs Matrix[complex128], iterations int) ([]complex128, [][]complex128) {
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
		return nil, nil
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
	outputs := make([][]complex128, inputs.Rows)
	for i := range outputs {
		outputs[i] = I.X[i*Width : (i+1)*Width]
	}
	return g.Set.ByName["g"].X, outputs
}

func main() {

}
