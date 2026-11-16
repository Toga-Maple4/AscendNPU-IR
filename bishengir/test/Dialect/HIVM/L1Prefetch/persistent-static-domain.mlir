// RUN: python3 %S/Inputs/check_persistent_static_domain.py bishengir-opt %S/persistent-runtime.mlir
// This intentionally narrows the previously unchecked fixed Persistent domain.
// Last-safe/first-overflow counts cover the comparison, row product and full
// tile byte tail. Accepted boundaries must reach transformation/launch metadata.
