// Copyright 2026 The QG Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"fmt"
	"io"
	"math"
	"math/cmplx"
	"os"
	//"runtime"
	//"sync/atomic"
)

const (
	// S is the scaling factor for the softmax
	S         = 1.0 - 1e-300
	InputSize = 256
)

// Number is a number
type Float interface {
	~float32 | ~float64 | ~complex64 | ~complex128
}

// Size is the size of the matrix
type Size struct {
	Name string
	Cols int
	Rows int
}

// Set is a set of sizes
type Set[T Float] struct {
	Sizes   []Size
	ByIndex []Matrix[T]
	ByName  map[string]*Matrix[T]
}

// Matrix is a float64 matrix
type Matrix[T Float] struct {
	Size
	Data []T
}

// NewMatrix creates a new float64 matrix
func NewMatrix[T Float](cols, rows int, data ...T) Matrix[T] {
	if data == nil {
		data = make([]T, 0, cols*rows)
	}
	return Matrix[T]{
		Size: Size{
			Cols: cols,
			Rows: rows,
		},
		Data: data,
	}
}

// NewMatrices creates matrices from a set
func NewMatrices[T Float](set Set[T], weights []T) Set[T] {
	offset := 0
	set.ByIndex = make([]Matrix[T], len(set.Sizes))
	set.ByName = make(map[string]*Matrix[T])
	for i, size := range set.Sizes {
		end := size.Cols * size.Rows
		set.ByIndex[i] = NewMatrix(size.Cols, size.Rows, weights[offset:offset+end]...)
		switch data := any(set.ByIndex[i].Data).(type) {
		case []float32:
			for ii := range data {
				factor := float32(math.Sqrt(2.0 / float64(size.Cols)))
				data[ii] *= data[ii] * factor
			}
		case []float64:
			for ii := range data {
				factor := math.Sqrt(2.0 / float64(size.Cols))
				data[ii] *= data[ii] * factor
			}
		case []complex64:
			for ii := range data {
				factor := complex(float32(math.Sqrt(2.0/float64(size.Cols))), 0)
				data[ii] *= data[ii] * factor
			}
		case []complex128:
			for ii := range data {
				factor := complex(math.Sqrt(2.0/float64(size.Cols)), 0)
				data[ii] *= data[ii] * factor
			}
		}
		set.ByName[size.Name] = &set.ByIndex[i]
		offset += end
	}
	return set
}

// Size is the size of the weight vector
func (s Set[T]) Size() int {
	sum := 0
	for _, size := range s.Sizes {
		sum += size.Cols * size.Rows
	}
	return sum
}

// Named returns the matrix named name
func (s Set[T]) Named(name string) Matrix[T] {
	return *s.ByName[name]
}

// MulT multiplies two matrices and computes the transpose
func (m Matrix[T]) MulT(n Matrix[T]) Matrix[T] {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols
	o := Matrix[T]{
		Size: Size{
			Cols: m.Rows,
			Rows: n.Rows,
		},
		Data: make([]T, 0, m.Rows*n.Rows),
	}
	lenn, lenm := len(n.Data), len(m.Data)
	for i := 0; i < lenn; i += columns {
		nn := n.Data[i : i+columns]
		for j := 0; j < lenm; j += columns {
			mm := m.Data[j : j+columns]
			o.Data = append(o.Data, dot(mm, nn))
		}
	}
	return o
}

func (m Matrix[T]) Tensor(n Matrix[T]) Matrix[T] {
	rows, cols := m.Cols, n.Cols
	if m.Rows != 1 || n.Rows != 1 {
		panic("number of rows should be 1")
	}
	output := NewMatrix(cols, rows, make([]T, cols*rows)...)
	for i := range rows {
		for ii := range cols {
			output.Data[i*cols+ii] = m.Data[i] * n.Data[ii]
		}
	}
	return output
}

// Add adds two float64 matrices
func (m Matrix[T]) Add(n Matrix[T]) Matrix[T] {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix[T]{
		Size: Size{
			Cols: m.Cols,
			Rows: m.Rows,
		},
		Data: make([]T, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value+n.Data[i%lenb])
	}
	return o
}

// Sub subtracts two float64 matrices
func (m Matrix[T]) Sub(n Matrix[T]) Matrix[T] {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix[T]{
		Size: Size{
			Cols: m.Cols,
			Rows: m.Rows,
		},
		Data: make([]T, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value-n.Data[i%lenb])
	}
	return o
}

// Hadamard multiples two float64 matrices
func (m Matrix[T]) Hadamard(n Matrix[T]) Matrix[T] {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix[T]{
		Size: Size{
			Cols: m.Cols,
			Rows: m.Rows,
		},
		Data: make([]T, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value*n.Data[i%lenb])
	}
	return o
}

// Softmax calculates the softmax of the matrix rows
func (m Matrix[T]) Softmax(t T) Matrix[T] {
	switch t := any(t).(type) {
	case float32:
		switch data := any(m.Data).(type) {
		case []float32:
			max := float32(0.0)
			for _, v := range data {
				v /= t
				if v > max {
					max = v
				}
			}
			s := max * S
			output := NewMatrix[T](m.Cols, m.Rows, make([]T, len(data))...)
			switch out := any(output.Data).(type) {
			case []float32:
				index := 0
				for i := 0; i < len(data); i += m.Cols {
					sum := float32(0.0)
					values := make([]float32, m.Cols)
					for j, value := range data[i : i+m.Cols] {
						values[j] = float32(math.Exp(float64(value/t - s)))
						sum += values[j]
					}
					for _, value := range values {
						out[index] = value / sum
						index++
					}
				}
			}
			return output
		}
	case float64:
		switch data := any(m.Data).(type) {
		case []float64:
			max := float64(0.0)
			for _, v := range data {
				v /= t
				if v > max {
					max = v
				}
			}
			s := max * S
			output := NewMatrix[T](m.Cols, m.Rows, make([]T, len(data))...)
			switch out := any(output.Data).(type) {
			case []float64:
				index := 0
				for i := 0; i < len(data); i += m.Cols {
					sum := float64(0.0)
					values := make([]float64, m.Cols)
					for j, value := range data[i : i+m.Cols] {
						values[j] = float64(math.Exp(float64(value/t - s)))
						sum += values[j]
					}
					for _, value := range values {
						out[index] = value / sum
						index++
					}
				}
			}
			return output
		}
	case complex64:
		switch data := any(m.Data).(type) {
		case []complex64:
			max := float32(0.0)
			for _, v := range data {
				v /= t
				if v := float32(cmplx.Abs(complex128(v))); v > max {
					max = v
				}
			}
			s := complex(float32(max*S), 0)
			output := NewMatrix[T](m.Cols, m.Rows, make([]T, len(data))...)
			switch out := any(output.Data).(type) {
			case []complex64:
				index := 0
				for i := 0; i < len(data); i += m.Cols {
					sum := complex(float32(0.0), 0)
					values := make([]complex64, m.Cols)
					for j, value := range data[i : i+m.Cols] {
						values[j] = complex64(cmplx.Exp(complex128(value/t - s)))
						sum += values[j]
					}
					for _, value := range values {
						out[index] = value / sum
						index++
					}
				}
			}
			return output
		}
	case complex128:
		switch data := any(m.Data).(type) {
		case []complex128:
			max := float32(0.0)
			for _, v := range data {
				v /= t
				if v := float32(cmplx.Abs(v)); v > max {
					max = v
				}
			}
			s := complex(float64(max*S), 0)
			output := NewMatrix[T](m.Cols, m.Rows, make([]T, len(data))...)
			switch out := any(output.Data).(type) {
			case []complex128:
				index := 0
				for i := 0; i < len(data); i += m.Cols {
					sum := complex(float64(0.0), 0)
					values := make([]complex128, m.Cols)
					for j, value := range data[i : i+m.Cols] {
						values[j] = cmplx.Exp(value/t - s)
						sum += values[j]
					}
					for _, value := range values {
						out[index] = value / sum
						index++
					}
				}
			}
			return output
		}
	}
	return Matrix[T]{}
}

// Sigmoid computes the sigmoid of a matrix
func (m Matrix[T]) Sigmoid() Matrix[T] {
	o := Matrix[T]{
		Size: Size{
			Cols: m.Cols,
			Rows: m.Rows,
		},
		Data: make([]T, m.Cols*m.Rows),
	}
	switch data := any(m.Data).(type) {
	case []float32:
		switch out := any(o.Data).(type) {
		case []float32:
			for i, value := range data {
				out[i] = float32(1 / (1 + math.Exp(float64(-value))))
			}
		}
	case []float64:
		switch out := any(o.Data).(type) {
		case []float64:
			for i, value := range data {
				out[i] = 1 / (1 + math.Exp(float64(-value)))
			}
		}
	case []complex64:
		switch out := any(o.Data).(type) {
		case []complex64:
			for i, value := range data {
				out[i] = complex64(1 / (1 + cmplx.Exp(complex128(-value))))
			}
		}
	case []complex128:
		switch out := any(o.Data).(type) {
		case []complex128:
			for i, value := range data {
				out[i] = 1 / (1 + cmplx.Exp(-value))
			}
		}
	}
	return o
}

// ReLu is the ramp function
/*func (m Matrix[T]) ReLu() Matrix[T] {
	o := Matrix[T]{
		Size: Size{
			Cols: m.Cols,
			Rows: m.Rows,
		},
		Data: make([]T, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		if value < 0 {
			value = 0
		}
		o.Data = append(o.Data, value)
	}
	return o
}*/

// Everett is the everett activation function
/*func (m Matrix[T]) Everett() Matrix[T] {
	o := Matrix[T]{
		Size: Size{
			Cols: 2 * m.Cols,
			Rows: m.Rows,
		},
		Data: make([]T, 0, 2*m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		positive := value
		if positive < 0 {
			positive = 0
		}
		o.Data = append(o.Data, positive)
		negative := value
		if negative > 0 {
			negative = 0
		}
		o.Data = append(o.Data, negative)
	}
	return o
}*/

// Entropy calculates the entropy of the matrix rows
/*func (m Matrix[T]) Entropy() Matrix[T] {
	output := NewMatrix[T](m.Rows, 1)
	for i := 0; i < len(m.Data); i += m.Cols {
		entropy := T(0.0)
		for _, value := range m.Data[i : i+m.Cols] {
			entropy += value * T(math.Log(float64(value)))
		}
		output.Data = append(output.Data, -entropy)
	}
	return output
}*/

// Sum sums the rows of a matrix
func (m Matrix[T]) Sum() Matrix[T] {
	o := Matrix[T]{
		Size: Size{
			Cols: m.Cols,
			Rows: 1,
		},
		Data: make([]T, m.Cols),
	}
	for i := 0; i < m.Rows; i++ {
		offset := i * m.Cols
		for j := range o.Data {
			o.Data[j] += m.Data[offset+j]
		}
	}
	return o
}

// T tramsposes a matrix
func (m Matrix[T]) T() Matrix[T] {
	o := Matrix[T]{
		Size: Size{
			Cols: m.Rows,
			Rows: m.Cols,
		},
		Data: make([]T, 0, m.Cols*m.Rows),
	}
	for i := 0; i < m.Cols; i++ {
		for j := 0; j < m.Rows; j++ {
			o.Data = append(o.Data, m.Data[j*m.Cols+i])
		}
	}
	return o
}

// Write writes the matrix to a file
func (m Matrix[T]) Write(output *os.File) error {
	switch data := any(m.Data).(type) {
	case []float64:
		buffer64 := make([]byte, 8)
		for _, parameter := range data {
			bits := math.Float64bits(parameter)
			for i := range buffer64 {
				buffer64[i] = byte((bits >> (8 * i)) & 0xFF)
			}
			n, err := output.Write(buffer64)
			if err != nil {
				return err
			}
			if n != len(buffer64) {
				return errors.New("8 bytes should be been written")
			}
		}
	case []float32:
		buffer32 := make([]byte, 4)
		for _, parameter := range data {
			bits := math.Float32bits(parameter)
			for i := range buffer32 {
				buffer32[i] = byte((bits >> (8 * i)) & 0xFF)
			}
			n, err := output.Write(buffer32)
			if err != nil {
				return err
			}
			if n != len(buffer32) {
				return errors.New("8 bytes should be been written")
			}
		}
	}
	return nil
}

// Read reads the matrix from a file
func (m *Matrix[T]) Read(input *os.File) error {
	switch any(m.Data).(type) {
	case []float64:
		buffer64 := make([]byte, 8)
		for range m.Rows {
			for range m.Cols {
				n, err := input.Read(buffer64)
				if err == io.EOF {
					return err
				} else if err != nil {
					return err
				}
				if n != len(buffer64) {
					return fmt.Errorf("not all bytes read: %d", n)
				}
				value := uint64(0)
				for k := 0; k < 8; k++ {
					value <<= 8
					value |= uint64(buffer64[7-k])
				}
				m.Data = append(m.Data, any(math.Float64frombits(value)).(T))
			}
		}
	case []float32:
		buffer32 := make([]byte, 4)
		for range m.Rows {
			for range m.Cols {
				n, err := input.Read(buffer32)
				if err == io.EOF {
					return err
				} else if err != nil {
					return err
				}
				if n != len(buffer32) {
					return fmt.Errorf("not all bytes read: %d", n)
				}
				value := uint32(0)
				for k := 0; k < 4; k++ {
					value <<= 8
					value |= uint32(buffer32[3-k])
				}
				m.Data = append(m.Data, any(math.Float32frombits(value)).(T))
			}
		}

	}
	return nil
}

func dot[T Float](x, y []T) (z T) {
	for i := range x {
		z += T(x[i] * y[i])
	}
	return z
}

func sqrt[T Float](x T) T {
	switch v := any(x).(type) {
	case float32:
		return any(float32(math.Sqrt(float64(v)))).(T)
	case float64:
		return any(math.Sqrt(v)).(T)
	case complex64:
		return any(complex64(cmplx.Sqrt(complex128(v)))).(T)
	case complex128:
		return any(cmplx.Sqrt(v)).(T)
	default:
		return x
	}
}

func softmax[T Float](values []T) {
	switch values := any(values).(type) {
	case []float32:
		max := float32(0.0)
		for _, v := range values {
			if v > max {
				max = v
			}
		}
		s := max * S
		sum := float32(0.0)
		for j, value := range values {
			values[j] = float32(math.Exp(float64(value - s)))
			sum += values[j]
		}
		for j, value := range values {
			values[j] = value / sum
		}
	case []float64:
		max := float64(0.0)
		for _, v := range values {
			if v > max {
				max = v
			}
		}
		s := max * S
		sum := 0.0
		for j, value := range values {
			values[j] = math.Exp(float64(value - s))
			sum += values[j]
		}
		for j, value := range values {
			values[j] = value / sum
		}
	case []complex64:
		max := float32(0.0)
		for _, v := range values {
			if v := float32(cmplx.Abs(complex128(v))); v > max {
				max = v
			}
		}
		s := complex(max*S, 0)
		sum := complex64(0.0)
		for j, value := range values {
			values[j] = complex64(cmplx.Exp(complex128(value - s)))
			sum += values[j]
		}
		for j, value := range values {
			values[j] = value / sum
		}
	case []complex128:
		max := float64(0.0)
		for _, v := range values {
			if v := cmplx.Abs(v); v > max {
				max = v
			}
		}
		s := complex(max*S, 0)
		sum := complex128(0.0)
		for j, value := range values {
			values[j] = cmplx.Exp(value - s)
			sum += values[j]
		}
		for j, value := range values {
			values[j] = value / sum
		}
	}
}

// CS implements cosine similarity
/*func (m Matrix[T]) CS(n Matrix[T]) T {
	var sum T
	var count T
	for i := 0; i < len(m.Data); i += m.Cols {
		md := m.Data[i : i+m.Cols]
		nd := n.Data[i : i+m.Cols]
		ab, aa, bb := dot(md, nd), dot(md, md), dot(nd, nd)
		if aa <= 0 || bb <= 0 {
			continue
		}
		sum += ab / (T(math.Sqrt(float64(aa))) * T(math.Sqrt(float64(bb))))
		count++
	}
	return sum / count
}*/

// CS implements cosine similarity
/*func (m Matrix[T]) Unit() Matrix[T] {
	o := NewMatrix[T](m.Cols, m.Rows)
	for i := 0; i < len(m.Data); i += m.Cols {
		md := m.Data[i : i+m.Cols]
		aa := dot(md, md)
		if aa <= 0 {
			for _, value := range md {
				o.Data = append(o.Data, value)
			}
			continue
		}
		aa = T(math.Sqrt(float64(aa)))
		for _, value := range md {
			o.Data = append(o.Data, value/aa)
		}
	}
	return o
}*/

// SelfAttention computes the self attention of Q, K, V
func SelfAttention[T Float](Q, K, V Matrix[T]) Matrix[T] {
	o := Matrix[T]{
		Size: Size{
			Cols: V.Cols,
			Rows: K.Rows,
		},
		Data: make([]T, 0, V.Rows*K.Rows),
	}
	outputs, values := make([]T, V.Cols), make([]T, Q.Rows)
	V = V.T()
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			values[j] = dot(K, Q)
		}
		softmax(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			outputs[j] = dot(values, V)
		}
		o.Data = append(o.Data, outputs...)
	}
	return o
}

// SelfAttention computes the self attention of Q, K, V
func selfAttention[T Float](input Matrix[T]) [InputSize]T {
	values := make([]T, input.Rows)
	V := input.T()
	output := [InputSize]T{}
	for i := 0; i < input.Rows; i++ {
		K := input.Data[i*input.Cols : (i+1)*input.Cols]
		for j := 0; j < input.Rows; j++ {
			Q := input.Data[j*input.Cols : (j+1)*input.Cols]
			values[j] = dot(K, Q)
		}
		softmax(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			output[j] += dot(values, V)
		}
	}
	aa := sqrt(dot(output[:], output[:]))
	for i, v := range output {
		output[i] = v / aa
	}
	return output
}

// RNG is a random number generator
type RNG uint32

// LFSRMask is a LFSR mask with a maximum period
const LFSRMask = 0x80000057

// Next returns the next random number
func (r *RNG) Next() uint32 {
	lfsr := *r
	lfsr = (lfsr >> 1) ^ (-(lfsr & 1) & LFSRMask)
	*r = lfsr
	return uint32(lfsr)
}

// Float32 returns a uniform float32
func (r *RNG) Float32() float32 {
	return float32(r.Next()) / float32(math.MaxUint32)
}

// Intn return a uniform random number less than n
func (r *RNG) Intn(n int) int {
	max := uint32((1 << 32) - 1 - (1<<32)%uint64(n))
	v := r.Next()
	for v > max {
		v = r.Next()
	}
	return int(v % uint32(n))
}

// PageRank is a counting based pagerank implementation
/*func PageRank[T Float](a float32, e int, seed uint32, adj Matrix[T]) Matrix[T] {
	for i := range adj.Rows {
		var sum T
		for ii := range adj.Cols {
			value := adj.Data[i*adj.Cols+ii]
			if value < 0 {
				value = -value
			}
			sum += value
		}
		for ii := range adj.Cols {
			adj.Data[i*adj.Cols+ii] /= sum
		}
	}
	counts := make([]int64, adj.Cols)
	iterations := adj.Rows * adj.Cols
	done := make(chan bool, 8)
	process := func(seed uint32) {
		rng := RNG(seed)
		node := rng.Intn(adj.Cols)
		for range iterations {
			if rng.Float32() > a {
				node = rng.Intn(adj.Cols)
			}
			total, selected, found, negative := T(0.0), T(rng.Float32()), false, false
			for i, weight := range adj.Data[node*adj.Cols : (node+1)*adj.Cols] {
				total += weight
				if weight < 0 {
					negative = true
					weight = -weight
				} else {
					negative = false
				}
				if selected < total {
					node, found = i, true
					break
				}
			}
			if !found {
				node = rng.Intn(adj.Cols)
			}
			counter := &counts[node]
			if negative {
				atomic.AddInt64(counter, -1)
			} else {
				atomic.AddInt64(counter, 1)
			}
		}
		done <- true
	}

	rng := RNG(seed)
	index, flights, cpus := 0, 0, runtime.NumCPU()
	for index < e && flights < cpus {
		go process(rng.Next())
		index++
		flights++
	}
	for index < e {
		<-done
		flights--

		go process(rng.Next())
		index++
		flights++
	}
	for range flights {
		<-done
	}

	sum := int64(0)
	for _, value := range counts {
		if value < 0 {
			value = -value
		}
		sum += value
	}
	p := NewMatrix[T](len(counts), 1)
	for _, value := range counts {
		p.Data = append(p.Data, T(value)/T(sum))
	}
	return p
}*/

// PageRankMarkov is a counting  pagerank implementation with a markov model
/*func PageRankMarkov[T Float](a float32, e int, seed uint32, adj Matrix[T]) Matrix[T] {
	for i := range adj.Rows {
		var sum T
		for ii := range adj.Cols {
			value := adj.Data[i*adj.Cols+ii]
			if value < 0 {
				value = -value
			}
			sum += value
		}
		for ii := range adj.Cols {
			adj.Data[i*adj.Cols+ii] /= sum
		}
	}
	rng := RNG(seed)
	counts := make([]int64, adj.Cols*adj.Rows)
	iterations := adj.Rows * adj.Cols
	done := make(chan bool, 8)
	process := func(seed uint32) {
		rng, node, prev := RNG(seed), rng.Intn(adj.Cols), 0
		for range iterations {
			if rng.Float32() > a {
				prev, node = node, rng.Intn(adj.Cols)
			}
			total, selected, found, negative := T(0.0), T(rng.Float32()), false, false
			for i, weight := range adj.Data[node*adj.Cols : (node+1)*adj.Cols] {
				if weight < 0 {
					negative = true
					weight = -weight
				} else {
					negative = false
				}
				total += weight
				if selected < total {
					prev, node, found = node, i, true
					break
				}
			}
			if !found {
				prev, node = node, rng.Intn(adj.Cols)
			}
			counter := &counts[node*adj.Cols+prev]
			if negative {
				atomic.AddInt64(counter, -1)
			} else {
				atomic.AddInt64(counter, 1)
			}
		}
		done <- true
	}

	index, flights, cpus := 0, 0, runtime.NumCPU()
	for index < e && flights < cpus {
		go process(rng.Next())
		index++
		flights++
	}
	for index < e {
		<-done
		flights--

		go process(rng.Next())
		index++
		flights++
	}
	for range flights {
		<-done
	}

	p := NewMatrix[T](adj.Cols, adj.Rows)
	for i := 0; i < p.Cols*p.Rows; i += p.Cols {
		sum := int64(0)
		for _, value := range counts[i : i+p.Cols] {
			if value < 0 {
				value = -value
			}
			sum += value
		}
		for _, value := range counts[i : i+p.Cols] {
			p.Data = append(p.Data, T(value)/T(sum))
		}
	}
	return p
}*/

// Transformer implements transform inference
/*func Transformer[T Float](set Set[T], inputs, outputs Matrix[T]) Matrix[T] {
	itags := set.Named("itags")
	if inputs.Rows != itags.Rows {
		panic("rows should be the same")
	}
	in := NewMatrix[T](itags.Cols+inputs.Cols, inputs.Rows)
	for i := 0; i < itags.Rows; i++ {
		in.Data = append(in.Data, itags.Data[i*itags.Cols:(i+1)*itags.Cols]...)
		in.Data = append(in.Data, inputs.Data[i*inputs.Cols:(i+1)*inputs.Cols]...)
	}
	otags := set.Named("otags")
	if outputs.Rows != otags.Rows {
		panic("rows should be the same")
	}
	out := NewMatrix[T](otags.Cols+outputs.Cols, outputs.Rows)
	for i := 0; i < otags.Rows; i++ {
		out.Data = append(out.Data, otags.Data[i*otags.Cols:(i+1)*otags.Cols]...)
		out.Data = append(out.Data, outputs.Data[i*outputs.Cols:(i+1)*outputs.Cols]...)
	}
	embeddingIn := set.Named("lembeddingIn").MulT(in).Add(set.Named("bembeddingIn")).ReLu()
	formIn := SelfAttention(set.Named("inQ").MulT(embeddingIn),
		set.Named("inK").MulT(embeddingIn),
		set.Named("inV").MulT(embeddingIn)).
		Add(embeddingIn)
	l1In := set.Named("l1In").MulT(formIn).Add(set.Named("b1In")).ReLu().Add(formIn)

	embeddingOut := set.Named("lembeddingOut").MulT(out).Add(set.Named("bembeddingOut")).ReLu()
	formOut := SelfAttention(set.Named("outQ1").MulT(embeddingOut),
		set.Named("outK1").MulT(embeddingOut),
		set.Named("outV1").MulT(embeddingOut)).
		Add(embeddingOut)
	formOut1 := SelfAttention(set.Named("outQ2").MulT(formOut),
		set.Named("outK2").MulT(l1In),
		set.Named("outV2").MulT(l1In)).
		Add(formOut)
	l1Out := set.Named("l1Out").MulT(formOut1).Add(set.Named("b1Out")).ReLu().Add(formOut1)
	return set.Named("linear").MulT(l1Out).Softmax(1)
}*/

// GramSchmidt performs the Gram-Schmidt process on the columns of a matrix
/*func (m Matrix[T]) GramSchmidt() Matrix[T] {
	numRows := m.Rows
	if numRows == 0 {
		return NewMatrix[T](0, 0)
	}
	numCols := m.Cols

	orthonormalBasis := make([][]T, numRows)
	for i := range orthonormalBasis {
		orthonormalBasis[i] = make([]T, numCols)
	}

	for j := 0; j < numCols; j++ {
		currentVector := NewMatrix(numRows, 1, make([]T, numRows)...)
		for i := 0; i < numRows; i++ {
			currentVector.Data[i] = m.Data[i*m.Cols+j] // Get the j-th column
		}

		u := currentVector

		// Subtract projections onto previously orthogonalized vectors
		for k := 0; k < j; k++ {
			qk := NewMatrix(numRows, 1, make([]T, numRows)...)
			for i := 0; i < numRows; i++ {
				qk.Data[i] = orthonormalBasis[i][k]
			}

			projection := qk.Hadamard(currentVector.MulT(qk))
			u = u.Sub(projection)
		}

		// Normalize the resulting vector
		normU := T(math.Sqrt(float64(u.MulT(u).Data[0])))
		if normU == 0 {
			// Handle linearly dependent vectors (e.g., set to zero vector or skip)
			// For simplicity, we'll just set it to a zero vector here.
			for i := range u.Data {
				orthonormalBasis[i][j] = 0
			}
		} else {
			for i := range u.Data {
				orthonormalBasis[i][j] = u.Data[i] / normU
			}
		}
	}

	n := NewMatrix(numCols, numRows, make([]T, numCols*numRows)...)
	for i := range orthonormalBasis {
		for ii := range orthonormalBasis[i] {
			n.Data[i*n.Cols+ii] = orthonormalBasis[i][ii]
		}
	}
	return n
}*/

// CS is cosine similarity
func CS[T Float](a []T, b []T) T {
	return dot(a, b)
}

// NCS is normalized cosine similarity
/*func NCS[T Float](a []T, b []T) T {
	aa, bb, ab := dot(a, a), dot(b, b), dot(a, b)
	if aa <= 0 {
		return 0
	}
	if bb <= 0 {
		return 0
	}
	return ab / (sqrt(aa) * sqrt(bb))
}*/
