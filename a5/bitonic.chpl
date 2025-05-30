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

proc bitonicSort(ref A: [] real, lo: int, cnt: int, dir: bool) {
  if cnt > 1 {
    const k = cnt / 2;
    bitonicSort(A, lo, k, true);
    bitonicSort(A, lo + k, k, false);
    bitonicMerge(A, lo, cnt, dir);
  }
}

proc main() {
  fillArray();

  var t: stopwatch;
  if timing then t.start();

  bitonicSort(A, 0, n, true);

  if timing then {
    t.stop();
    writeln("Bitonic sort time: ", t.elapsed());
  }

  if verbose > 0 then writeln("Sorted A[0..", verbose-1, "]: ", A[0..#verbose]);
}
