// RUN: triton-opt %s -allow-unregistered-dialect -split-input-file -tritonamdgpu-canonicalize-pointers="enable-large-tensor-ptr-canon=false" -canonicalize -verify-diagnostics | FileCheck %s

// this case is copied from amd-canonicalize-pointers-no-large-tensor.mlir. With
// enable-large-tensor-ptr-canon=false, the input is not changed at all.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @conversion1(%arg0: !tt.ptr<f32>) -> tensor<1024xf32> {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.splat %1 : i32 -> tensor<1024xi32>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %4 = tt.addptr %3, %2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %5 = tt.load %4 : tensor<1024x!tt.ptr<f32>>
    tt.return %5 : tensor<1024xf32>
  }
}

// CHECK-LABEL:   tt.func @conversion1
// CHECK: %[[ADDPTR:.*]] = tt.addptr
// CHECK:                = tt.load %[[ADDPTR]]

// ---
// Verify that a scalar select no longer crashes
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: _scalar_select
  tt.func public @_scalar_select(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg4: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg5: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c9_i32 = arith.constant 9 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = tt.addptr %arg3, %0 : !tt.ptr<i32>, i32
    %4 = tt.load %3 : !tt.ptr<i32>
    %5 = arith.addi %1, %4 : i32
    %6 = arith.addi %0, %c1_i32 : i32
    %7 = tt.addptr %arg3, %6 : !tt.ptr<i32>, i32
    %8 = tt.load %7 : !tt.ptr<i32>
    %9 = arith.cmpi sge, %2, %c9_i32 : i32
    %10 = tt.addptr %arg0, %5 : !tt.ptr<bf16>, i32
    %11 = arith.muli %5, %arg8 : i32
    %12 = arith.muli %2, %arg9 : i32
    %13 = arith.addi %11, %12 : i32
    %14 = tt.addptr %arg1, %13 : !tt.ptr<bf16>, i32
    %15 = tt.addptr %arg4, %0 : !tt.ptr<i32>, i32
    %16 = tt.load %15 : !tt.ptr<i32>
    %17 = tt.addptr %arg5, %0 : !tt.ptr<i32>, i32
    %18 = tt.load %17 : !tt.ptr<i32>
    %19 = arith.addi %16, %18 : i32
    %20 = arith.subi %8, %5 : i32
    %21 = arith.subi %19, %20 : i32
    %22 = arith.subi %2, %c9_i32 : i32
    %23 = arith.muli %22, %arg7 : i32
    %24 = arith.muli %21, %arg6 : i32
    %25 = arith.addi %23, %24 : i32
    %26 = tt.addptr %arg2, %25 : !tt.ptr<bf16>, i32
    // CHECK-COUNT-2: tt.addptr
    // CHECK: arith.select
    %27 = arith.select %9, %26, %14 : !tt.ptr<bf16>
    %28 = tt.load %10 : !tt.ptr<bf16>
    tt.store %27, %28 : !tt.ptr<bf16>
    tt.return
  }
}

// -----

// An access assertion overrides ordinary S specialization, including for the
// unselected small pointer at this join. Preserve i32 overflow before sext and
// accumulate separate pointer additions in i64.
module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @access_range_mixed_join
  tt.func @access_range_mixed_join(%a: !tt.ptr<i8> {tt.pointer_access_range = 32 : i32, tt.pointer_range = 32 : i32}, %b: !tt.ptr<i8> {tt.pointer_range = 32 : i32}, %x: i32, %y: i32, %cond: i1) -> i64 {
    // CHECK: %[[WRAP:.*]] = arith.addi %arg2, %arg3 : i32
    // CHECK: %[[EXT:.*]] = arith.extsi %[[WRAP]] : i32 to i64
    // CHECK: %[[YEXT:.*]] = arith.extsi %arg3 : i32 to i64
    // CHECK: %[[SUM:.*]] = arith.addi %[[EXT]], %[[YEXT]] : i64
    // CHECK: %[[BASE:.*]] = arith.select %arg4, %arg0, %arg1 : !tt.ptr<i8>
    // CHECK: %[[OFFSET:.*]] = arith.select %arg4, %[[SUM]], %{{.*}} : i64
    // CHECK: %[[PTR:.*]] = tt.addptr %[[BASE]], %[[OFFSET]] : !tt.ptr<i8>, i64
    // CHECK: tt.ptr_to_int %[[PTR]]
    %wrapped = arith.addi %x, %y : i32
    %ap = tt.addptr %a, %wrapped : !tt.ptr<i8>, i32
    %aq = tt.addptr %ap, %y : !tt.ptr<i8>, i32
    %bp = tt.addptr %b, %wrapped : !tt.ptr<i8>, i32
    %bq = tt.addptr %bp, %y : !tt.ptr<i8>, i32
    %p = arith.select %cond, %aq, %bq : !tt.ptr<i8>
    %i = tt.ptr_to_int %p : !tt.ptr<i8> -> i64
    tt.return %i : i64
  }
}

// -----

// Unsupported pointer forms retain the entire original pointer expression.
module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @access_range_trans_fallback
  // CHECK: tt.addptr {{.*}} tensor<2x2xi32>
  // CHECK: tt.trans
  // CHECK: tt.ptr_to_int
  // CHECK-NOT: arith.extsi
  tt.func @access_range_trans_fallback(%a: !tt.ptr<i8> {tt.pointer_access_range = 32 : i32, tt.pointer_range = 32 : i32}, %offset: tensor<2x2xi32>) -> tensor<2x2xi64> {
    %base = tt.splat %a : !tt.ptr<i8> -> tensor<2x2x!tt.ptr<i8>>
    %p = tt.addptr %base, %offset : tensor<2x2x!tt.ptr<i8>>, tensor<2x2xi32>
    %t = tt.trans %p {order = array<i32: 1, 0>} : tensor<2x2x!tt.ptr<i8>> -> tensor<2x2x!tt.ptr<i8>>
    %i = tt.ptr_to_int %t : tensor<2x2x!tt.ptr<i8>> -> tensor<2x2xi64>
    tt.return %i : tensor<2x2xi64>
  }
}

// -----

// A while carrying ordinary bool state retains its original pointer form.
module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @access_range_bool_while_fallback
  // CHECK: scf.while {{.*}} : (!tt.ptr<i8>, i1) -> !tt.ptr<i8>
  // CHECK: scf.condition
  // CHECK: tt.addptr {{.*}} !tt.ptr<i8>, i32
  tt.func @access_range_bool_while_fallback(%a: !tt.ptr<i8> {tt.pointer_access_range = 32 : i32}, %cond: i1) -> i64 {
    %false = arith.constant false
    %one = arith.constant 1 : i32
    %p, %c = scf.while (%pointer = %a, %live = %cond) : (!tt.ptr<i8>, i1) -> (!tt.ptr<i8>, i1) {
      scf.condition(%live) %pointer, %live : !tt.ptr<i8>, i1
    } do {
    ^bb0(%pointer: !tt.ptr<i8>, %live: i1):
      %next = tt.addptr %pointer, %one : !tt.ptr<i8>, i32
      scf.yield %next, %false : !tt.ptr<i8>, i1
    }
    %i = tt.ptr_to_int %p : !tt.ptr<i8> -> i64
    tt.return %i : i64
  }
}

// -----

// SCF permits different input and result signatures. The pointer pass must
// leave this loop intact instead of replacing its one result with two values.
module attributes {"ttg.num-warps" = 4} {
  // CHECK-LABEL: @access_range_while_different_signatures
  // CHECK: scf.while {{.*}} : (!tt.ptr<i8>, i32) -> !tt.ptr<i8>
  // CHECK: tt.addptr {{.*}} !tt.ptr<i8>, i32
  tt.func @access_range_while_different_signatures(%a: !tt.ptr<i8> {tt.pointer_access_range = 32 : i32}) -> i64 {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %p = scf.while (%pointer = %a, %count = %zero) : (!tt.ptr<i8>, i32) -> !tt.ptr<i8> {
      %continue = arith.cmpi slt, %count, %one : i32
      scf.condition(%continue) %pointer : !tt.ptr<i8>
    } do {
    ^bb0(%pointer: !tt.ptr<i8>):
      %next = tt.addptr %pointer, %one : !tt.ptr<i8>, i32
      scf.yield %next, %one : !tt.ptr<i8>, i32
    }
    %i = tt.ptr_to_int %p : !tt.ptr<i8> -> i64
    tt.return %i : i64
  }
}

// -----

// An i1 loop that does not carry a pointer must not disable pointer
// canonicalization for an otherwise eligible pointer expression in the same
// function. This test checks canonicalization, not buffer-op conversion.
module attributes {"ttg.num-warps" = 4} {
  // CHECK-LABEL: @access_range_unrelated_bool_while
  // CHECK: scf.while
  // CHECK: arith.extsi {{.*}} : i32 to i64
  // CHECK: tt.addptr {{.*}} : !tt.ptr<i8>, i64
  tt.func @access_range_unrelated_bool_while(%a: !tt.ptr<i8> {tt.pointer_access_range = 32 : i32}, %x: i32, %live: i1) -> i64 {
    %false = arith.constant false
    %zero = arith.constant 0 : i32
    %done = scf.while (%flag = %live) : (i1) -> i1 {
      scf.condition(%flag) %flag : i1
    } do {
    ^bb0(%flag: i1):
      scf.yield %false : i1
    }
    %offset = arith.select %done, %x, %zero : i32
    %p = tt.addptr %a, %offset : !tt.ptr<i8>, i32
    %i = tt.ptr_to_int %p : !tt.ptr<i8> -> i64
    tt.return %i : i64
  }
}

// -----

// A while can also create a pointer result from integer init state. Check both
// sides of its signature before expanding any pointer into a pair.
module attributes {"ttg.num-warps" = 4} {
  // CHECK-LABEL: @access_range_while_pointer_result
  // CHECK: scf.while {{.*}} : (i32) -> !tt.ptr<i8>
  // CHECK: tt.addptr {{.*}} !tt.ptr<i8>, i32
  tt.func @access_range_while_pointer_result(%a: !tt.ptr<i8> {tt.pointer_access_range = 32 : i32}) -> i64 {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %p = scf.while (%count = %zero) : (i32) -> !tt.ptr<i8> {
      %continue = arith.cmpi slt, %count, %one : i32
      %next = tt.addptr %a, %count : !tt.ptr<i8>, i32
      scf.condition(%continue) %next : !tt.ptr<i8>
    } do {
    ^bb0(%pointer: !tt.ptr<i8>):
      scf.yield %one : i32
    }
    %i = tt.ptr_to_int %p : !tt.ptr<i8> -> i64
    tt.return %i : i64
  }
}
