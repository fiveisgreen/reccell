import torch
print(torch.__version__) 
# 1.1.0

data = torch.load('issue/I04_s1_05.pt') # also E15_s1_08, D14_s2_03

data.type()                                                                                                           
# 'torch.FloatTensor'

torch.save(data.type(torch.uint8),'issue/I04_test.pt')                                                                

data = torch.load('issue/I04_test.pt') 
# error message
"""
ValueError                                Traceback (most recent call last)
~/anaconda3/envs/tf/lib/python3.7/tarfile.py in nti(s)
    186             s = nts(s, "ascii", "strict")
--> 187             n = int(s.strip() or "0", 8)
    188         except ValueError:

ValueError: invalid literal for int() with base 8: 'tils\n_re'

During handling of the above exception, another exception occurred:

InvalidHeaderError                        Traceback (most recent call last)
~/anaconda3/envs/tf/lib/python3.7/tarfile.py in next(self)
   2288             try:
-> 2289                 tarinfo = self.tarinfo.fromtarfile(self)
   2290             except EOFHeaderError as e:

~/anaconda3/envs/tf/lib/python3.7/tarfile.py in fromtarfile(cls, tarfile)
   1094         buf = tarfile.fileobj.read(BLOCKSIZE)
-> 1095         obj = cls.frombuf(buf, tarfile.encoding, tarfile.errors)
   1096         obj.offset = tarfile.fileobj.tell() - BLOCKSIZE

~/anaconda3/envs/tf/lib/python3.7/tarfile.py in frombuf(cls, buf, encoding, errors)
   1036 
-> 1037         chksum = nti(buf[148:156])
   1038         if chksum not in calc_chksums(buf):

~/anaconda3/envs/tf/lib/python3.7/tarfile.py in nti(s)
    188         except ValueError:
--> 189             raise InvalidHeaderError("invalid header")
    190     return n

InvalidHeaderError: invalid header

During handling of the above exception, another exception occurred:

ReadError                                 Traceback (most recent call last)
~/anaconda3/envs/tf/lib/python3.7/site-packages/torch/serialization.py in _load(f, map_location, pickle_module, **pickle_load_args)
    555         try:
--> 556             return legacy_load(f)
    557         except tarfile.TarError:

~/anaconda3/envs/tf/lib/python3.7/site-packages/torch/serialization.py in legacy_load(f)
    466 
--> 467         with closing(tarfile.open(fileobj=f, mode='r:', format=tarfile.PAX_FORMAT)) as tar, \
    468                 mkdtemp() as tmpdir:

~/anaconda3/envs/tf/lib/python3.7/tarfile.py in open(cls, name, mode, fileobj, bufsize, **kwargs)
   1590                 raise CompressionError("unknown compression type %r" % comptype)
-> 1591             return func(name, filemode, fileobj, **kwargs)
   1592 

~/anaconda3/envs/tf/lib/python3.7/tarfile.py in taropen(cls, name, mode, fileobj, **kwargs)
   1620             raise ValueError("mode must be 'r', 'a', 'w' or 'x'")
-> 1621         return cls(name, mode, fileobj, **kwargs)
   1622 

~/anaconda3/envs/tf/lib/python3.7/tarfile.py in __init__(self, name, mode, fileobj, format, tarinfo, dereference, ignore_zeros, encoding, errors, pax_headers, debug, errorlevel, copybufsize)
   1483                 self.firstmember = None
-> 1484                 self.firstmember = self.next()
   1485 

~/anaconda3/envs/tf/lib/python3.7/tarfile.py in next(self)
   2300                 elif self.offset == 0:
-> 2301                     raise ReadError(str(e))
   2302             except EmptyHeaderError:

ReadError: invalid header

During handling of the above exception, another exception occurred:

RuntimeError                              Traceback (most recent call last)
<ipython-input-5-0dee5a4615af> in <module>
----> 1 data = torch.load('issue/I04_test.pt')

~/anaconda3/envs/tf/lib/python3.7/site-packages/torch/serialization.py in load(f, map_location, pickle_module, **pickle_load_args)
    385         f = f.open('rb')
    386     try:
--> 387         return _load(f, map_location, pickle_module, **pickle_load_args)
    388     finally:
    389         if new_fd:

~/anaconda3/envs/tf/lib/python3.7/site-packages/torch/serialization.py in _load(f, map_location, pickle_module, **pickle_load_args)
    558             if zipfile.is_zipfile(f):
    559                 # .zip is used for torch.jit.save and will throw an un-pickling error here
--> 560                 raise RuntimeError("{} is a zip archive (did you mean to use torch.jit.load()?)".format(f.name))
    561             # if not a tarfile, reset file offset and proceed
    562             f.seek(0)

RuntimeError: issue/I04_test.pt is a zip archive (did you mean to use torch.jit.load()?)
"""