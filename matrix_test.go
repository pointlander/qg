// Copyright 2026 The QG Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/cmplx"
	"testing"
)

type M[T Number] struct {
	V Math[T]
}

func (m M[T]) X() T {
	return m.V.Exp()
}

func BenchmarkExpControl(b *testing.B) {
	for i := 0; i < b.N; i++ {
		a := float32(1.0)
		math.Exp(float64(a))
		b := 1.0
		math.Exp(b)
		c := complex64(1.0)
		cmplx.Exp(complex128(c))
		d := complex128(1.0)
		cmplx.Exp(d)
	}
}

func BenchmarkExpDirect(b *testing.B) {
	for i := 0; i < b.N; i++ {
		a := F32(1.0)
		a.Exp()
		b := F64(1.0)
		b.Exp()
		c := C64(1.0)
		c.Exp()
		d := C128(1.0)
		d.Exp()
	}
}

func BenchmarkExpMatrix(bb *testing.B) {
	a := M[F32]{V: F32(1.0)}
	b := M[F64]{V: F64(1.0)}
	c := M[C64]{V: C64(1.0)}
	d := M[C128]{V: C128(1.0)}
	for i := 0; i < bb.N; i++ {
		_ = a.X() + a.V.(F32)
		_ = b.X() + b.V.(F64)
		_ = c.X() + c.V.(C64)
		_ = d.X() + d.V.(C128)
	}
}

func BenchmarkExpFloat(b *testing.B) {
	for i := 0; i < b.N; i++ {
		a := float32(1.0)
		exp(a)
		b := 1.0
		exp(b)
		c := complex64(1.0)
		exp(c)
		d := complex128(1.0)
		exp(d)
	}
}
