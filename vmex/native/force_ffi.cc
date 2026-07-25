#include <algorithm>
#include <array>
#include <cstdint>
#include <thread>
#include <type_traits>
#include <vector>

#include "pybind11/pybind11.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
namespace py = pybind11;

template <typename T>
ffi::Error Project(int64_t mpol, int64_t ntor, int64_t ntheta2,
                   int64_t thread_count, bool include_edge, bool asym,
                   ffi::AnyBuffer kernels, ffi::AnyBuffer cosmui,
                   ffi::AnyBuffer sinmui, ffi::AnyBuffer cosmumi,
                   ffi::AnyBuffer sinmumi, ffi::AnyBuffer cosnv,
                   ffi::AnyBuffer sinnv, ffi::AnyBuffer cosnvn,
                   ffi::AnyBuffer sinnvn, ffi::Result<ffi::AnyBuffer> result,
                   ffi::Result<ffi::AnyBuffer> scratch) {
  auto kd = kernels.dimensions();
  if (kd.size() != 4 || kd[0] != 20 || kd[2] < ntheta2) {
    return ffi::Error::InvalidArgument(
        "kernels must have shape (20, ns, ntheta3, nzeta)");
  }
  const int64_t ns = kd[1];
  const int64_t ntheta3 = kd[2];
  const int64_t nzeta = kd[3];
  const int64_t ncols = ntor + 1;
  const int64_t tasks = ns * mpol;
  const int64_t workers = std::max<int64_t>(
      1, std::min<int64_t>({thread_count, tasks, scratch->dimensions()[0]}));

  const T* k = kernels.typed_data<T>();
  const T* cm = cosmui.typed_data<T>();
  const T* sm = sinmui.typed_data<T>();
  const T* cmm = cosmumi.typed_data<T>();
  const T* smm = sinmumi.typed_data<T>();
  const T* cn = cosnv.typed_data<T>();
  const T* sn = sinnv.typed_data<T>();
  const T* cnn = cosnvn.typed_data<T>();
  const T* snn = sinnvn.typed_data<T>();
  T* out = result->typed_data<T>();
  T* tmp = scratch->typed_data<T>();

  auto kernel = [&](int64_t channel, int64_t parity, int64_t s, int64_t i,
                    int64_t z) -> T {
    const int64_t plane = 2 * channel + parity;
    return k[((plane * ns + s) * ntheta3 + i) * nzeta + z];
  };
  auto table = [mpol](const T* a, int64_t i, int64_t m) -> T {
    return a[i * mpol + m];
  };
  auto ztable = [ncols](const T* a, int64_t z, int64_t n) -> T {
    return a[z * ncols + n];
  };
  auto output = [=](int64_t channel, int64_t s, int64_t m,
                    int64_t n) -> int64_t {
    return ((channel * ns + s) * mpol + m) * ncols + n;
  };

  auto run = [&](int64_t worker) {
    T* w = tmp + worker * 12 * nzeta;
    for (int64_t task = worker; task < tasks; task += workers) {
      const int64_t s = task / mpol;
      const int64_t m = task % mpol;
      const int64_t p = m & 1;
      const T xmpq = static_cast<T>(m * (m - 1));
      std::fill(w, w + 12 * nzeta, static_cast<T>(0));
      for (int64_t i = 0; i < ntheta2; ++i) {
        const T c = table(cm, i, m);
        const T q = table(sm, i, m);
        const T dc = table(cmm, i, m);
        const T dq = table(smm, i, m);
        for (int64_t z = 0; z < nzeta; ++z) {
          const T r = kernel(0, p, s, i, z);
          const T ru = kernel(1, p, s, i, z);
          const T rv = kernel(2, p, s, i, z);
          const T zz = kernel(3, p, s, i, z);
          const T zu = kernel(4, p, s, i, z);
          const T zv = kernel(5, p, s, i, z);
          const T lu = kernel(6, p, s, i, z);
          const T lv = kernel(7, p, s, i, z);
          const T rc = kernel(8, p, s, i, z);
          const T zc = kernel(9, p, s, i, z);
          w[0 * nzeta + z] += r * c + ru * dq + xmpq * rc * c;
          w[1 * nzeta + z] -= rv * c;
          w[2 * nzeta + z] += r * q + ru * dc + xmpq * rc * q;
          w[3 * nzeta + z] -= rv * q;
          w[4 * nzeta + z] += zz * c + zu * dq + xmpq * zc * c;
          w[5 * nzeta + z] -= zv * c;
          w[6 * nzeta + z] += zz * q + zu * dc + xmpq * zc * q;
          w[7 * nzeta + z] -= zv * q;
          w[8 * nzeta + z] += lu * dq;
          w[9 * nzeta + z] -= lv * c;
          w[10 * nzeta + z] += lu * dc;
          w[11 * nzeta + z] -= lv * q;
        }
      }

      const std::array<int, 3> ca =
          asym ? std::array<int, 3>{2, 4, 8}
               : std::array<int, 3>{0, 6, 10};
      const std::array<int, 3> cb =
          asym ? std::array<int, 3>{3, 5, 9}
               : std::array<int, 3>{1, 7, 11};
      const std::array<int, 3> sa =
          asym ? std::array<int, 3>{0, 6, 10}
               : std::array<int, 3>{2, 4, 8};
      const std::array<int, 3> sb =
          asym ? std::array<int, 3>{1, 7, 11}
               : std::array<int, 3>{3, 5, 9};
      for (int64_t n = 0; n < ncols; ++n) {
        for (int family = 0; family < 3; ++family) {
          T co = 0;
          T si = 0;
          for (int64_t z = 0; z < nzeta; ++z) {
            co += w[ca[family] * nzeta + z] * ztable(cn, z, n);
            if (ntor > 0) {
              co += w[cb[family] * nzeta + z] * ztable(snn, z, n);
              si += w[sa[family] * nzeta + z] * ztable(sn, z, n) +
                    w[sb[family] * nzeta + z] * ztable(cnn, z, n);
            }
          }
          const bool rz = family < 2;
          const int64_t first = rz ? (m == 0 ? 0 : 1) : 1;
          const int64_t last = rz ? (include_edge ? ns - 1 : ns - 2) : ns - 1;
          const bool active = s >= first && s <= last;
          out[output(2 * family, s, m, n)] =
              active ? co : static_cast<T>(0);
          out[output(2 * family + 1, s, m, n)] =
              active ? si : static_cast<T>(0);
        }
      }
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(std::max<int64_t>(0, workers - 1));
  for (int64_t worker = 1; worker < workers; ++worker) {
    threads.emplace_back(run, worker);
  }
  run(0);
  for (auto& thread : threads) thread.join();
  return ffi::Error::Success();
}

ffi::Error Dispatch(int64_t mpol, int64_t ntor, int64_t ntheta2,
                    int64_t threads, bool include_edge, bool asym,
                    ffi::AnyBuffer kernels, ffi::AnyBuffer cosmui,
                    ffi::AnyBuffer sinmui, ffi::AnyBuffer cosmumi,
                    ffi::AnyBuffer sinmumi, ffi::AnyBuffer cosnv,
                    ffi::AnyBuffer sinnv, ffi::AnyBuffer cosnvn,
                    ffi::AnyBuffer sinnvn,
                    ffi::Result<ffi::AnyBuffer> result,
                    ffi::Result<ffi::AnyBuffer> scratch) {
  if (threads < 1) {
    return ffi::Error::InvalidArgument("threads must be positive");
  }
  switch (kernels.element_type()) {
    case ffi::F32:
      return Project<float>(mpol, ntor, ntheta2, threads, include_edge, asym,
                            kernels, cosmui, sinmui, cosmumi, sinmumi, cosnv,
                            sinnv, cosnvn, sinnvn, result, scratch);
    case ffi::F64:
      return Project<double>(mpol, ntor, ntheta2, threads, include_edge, asym,
                             kernels, cosmui, sinmui, cosmumi, sinmumi, cosnv,
                             sinnv, cosnvn, sinnvn, result, scratch);
    default:
      return ffi::Error::InvalidArgument("only float32 and float64 are supported");
  }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    VmexForceProjection, Dispatch,
    ffi::Ffi::Bind()
        .Attr<int64_t>("mpol")
        .Attr<int64_t>("ntor")
        .Attr<int64_t>("ntheta2")
        .Attr<int64_t>("threads")
        .Attr<bool>("include_edge")
        .Attr<bool>("asym")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

template <typename T>
py::capsule Encapsulate(T* fn) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error*, T, XLA_FFI_CallFrame*>);
  return py::capsule(reinterpret_cast<void*>(fn));
}

PYBIND11_MODULE(_force_ffi, module) {
  module.def("registrations", []() {
    py::dict registrations;
    registrations["vmex_force_projection"] = Encapsulate(VmexForceProjection);
    return registrations;
  });
}
