use Time, Random, BlockDist;

config const n = 2**16;
config const timing = true;
config const verbose = 0;

const Space = {0..#n};
const D = Space dmapped Block(Space);
var A: [D] real;

proc fillArray() {
  fillRandom(A, 0.0, 100000.0);
  if verbose > 0 then writeln("Initial A[0..", verbose-1, "]: ", A[0..#verbose]);
}

proc quicksort(ref A: [] real, lo: int, hi: int) {
  if lo < hi {
    const p = partition(A, lo, hi);
    quicksort(A, lo, p - 1);
    quicksort(A, p + 1, hi);
  }
}

proc partition(ref A: [] real, lo: int, hi: int): int {
  const pivot = A[hi];
  var i = lo - 1;
  for j in lo..hi-1 do
    if A[j] <= pivot {
      i += 1;
      A[i] <=> A[j];
    }
  A[i+1] <=> A[hi];
  return i + 1;
}

proc compareAndSwap(ref A: [] real, i: int, j: int, dir: bool) {
  if dir == (A[i] > A[j]) then
    A[i] <=> A[j];
}

proc bitonicMerge(ref A: [] real, lo: int, cnt: int, dir: bool) {
  if cnt > 1 {
    const k = cnt / 2;
    forall i in lo..#k do
      compareAndSwap(A, i, i + k, dir);
    bitonicMerge(A, lo, k, dir);
    bitonicMerge(A, lo + k, k, dir);
  }
}

proc hybridSort(ref A: [D] real) {
  // Step 1: Local quicksort on each locale's portion
  forall loc in Locales do on loc {
    const myDom = A.localSubdomain();
    const lo = myDom.low;
    const hi = myDom.high;
    quicksort(A, lo, hi);
  }
  // Step 2: Global bitonic merge across locales
  bitonicMerge(A, 0, n, true);
}

proc main() {
  fillArray();

  var t: stopwatch;
  if timing then t.start();

  hybridSort(A);

  if timing then {
    t.stop();
    writeln("Hybrid sort time: ", t.elapsed());
  }

  if verbose > 0 then writeln("Sorted A[0..", verbose-1, "]: ", A[0..#verbose]);
}
