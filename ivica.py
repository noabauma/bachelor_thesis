import h5py
f = h5py.File('restart_files/pv.PV-00000.h5', 'r')
x = f['position'][()]
print(x.min(axis=0))
print(x.max(axis=0))