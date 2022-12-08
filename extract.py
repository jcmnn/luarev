import struct
import os
import ctypes
import pickle
import os.path


class OodleLZ_CompressOptions(ctypes.Structure):
    _fields_ = [('unused_was_verbosity', ctypes.c_uint32),
                ('minMatchLen', ctypes.c_int32),
                ('seekChunkReset', ctypes.c_bool),
                ('seekChunkLen', ctypes.c_int32),
                ('profile', ctypes.c_int32),
                ('dictionarySize', ctypes.c_int32),
                ('spaceSpeedTradeoffBytes', ctypes.c_int32),
                ('unused_was_maxHuffmansPerChunk', ctypes.c_int32),
                ('sendQuantumCRCs', ctypes.c_bool),
                ('maxLocalDictionarySize', ctypes.c_int32),
                ('makeLongRangeMatcher', ctypes.c_bool),
                ('matchTableSizeLog2', ctypes.c_int32),
                ('jobify', ctypes.c_int32),
                ('jobifyUserPtr', ctypes.c_void_p),
                ('farMatchMinLen', ctypes.c_int32),
                ('farMatchOffsetLog2', ctypes.c_int32),
                ('reserved', ctypes.c_uint32 * 4)]


oodle_dll = ctypes.cdll.LoadLibrary(
    "E:\\Warframe\\Downloaded\\Public\\Tools\\Oodle\\x64\\final\\oo2core_9_win64.dll")

oodleDecompressProto = ctypes.WINFUNCTYPE(
    ctypes.c_int64,  # Return type.

    ctypes.c_void_p,  # compBuf
    ctypes.c_int64,  # compBufSize
    ctypes.c_void_p,  # rawBuf
    ctypes.c_int64,  # rawLen
    ctypes.c_int32,  # fuzzSafe
    ctypes.c_int32,  # checkCRC
    ctypes.c_int32,  # verbosity
    ctypes.c_void_p,  # decBufBase
    ctypes.c_int64,  # decBufSize
    ctypes.c_void_p,  # fpCallback
    ctypes.c_void_p,  # callbackUserData
    ctypes.c_void_p,  # decoderMemory
    ctypes.c_int64,  # decoderMemorySize
    ctypes.c_int32  # threadPhase
)
decompressParams = (1, "compBuf", 0), (1, "compBufSize", 0), (1, "rawBuf", 0), (1, "rawLen", 0), (1, "fuzzSafe", 1), (1, "checkCRC", 0), (1, "verbosity", 0), (1,
                                                                                                                                                               "decBufBase", 0), (1, "decBufSize", 0), (1, "fpCallback", 0), (1, "callbackUserData", 0), (1, "decoderMemory", 0), (1, "decoderMemorySize", 0), (1, "threadPhase", 3),
decompress = oodleDecompressProto(
    ("OodleLZ_Decompress", oodle_dll), decompressParams)


oodleSetPrintfProto = ctypes.WINFUNCTYPE(
    ctypes.c_void_p,  # Return type.
    ctypes.c_void_p
)
setPrintfParams = (1, "fp_rrRawPrintf"),
setPrintf = oodleSetPrintfProto(
    ("OodleCore_Plugins_SetPrintf", oodle_dll), setPrintfParams)


OodleLZ_CompressOptions_GetDefault_proto = ctypes.WINFUNCTYPE(
    ctypes.POINTER(OodleLZ_CompressOptions),

    ctypes.c_int32,
    ctypes.c_int32
)
OodleLZ_CompressOptions_GetDefault_params = (
    1, "compressor", 1), (1, "izLevel", 4),
OodleLZ_CompressOptions_GetDefault = OodleLZ_CompressOptions_GetDefault_proto(
    ("OodleLZ_CompressOptions_GetDefault", oodle_dll), OodleLZ_CompressOptions_GetDefault_params)


OodleLZ_Compress_proto = ctypes.WINFUNCTYPE(
    ctypes.c_int32,

    ctypes.c_int32,
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int32
)
OodleLZ_Compress_params = (1, "compressor"), (1, "rawBuf"), (1, "rawLen"), (1, "compBuf"), (1, "level"), (
    1, "pOptions", None), (1, "dictionaryBase", None), (1, "lrm", None), (1, "scratchMem", None), (1, "scratchSize", 0),
OodleLZ_Compress = OodleLZ_Compress_proto(
    ("OodleLZ_Compress", oodle_dll), OodleLZ_Compress_params)


@ctypes.CFUNCTYPE(None, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32, ctypes.c_char_p)
def printf_cb(verboseLevel, file, line, fmt):
    print(f"{fmt}")

# setPrintf(printf_cb)
# print(setPrintf(oodle_dll["OodleCore_Plugin_Printf_Default"]))
# print(setPrintf(oodle_dll["OodleCore_Plugin_Printf_Verbose"]))

"""
with open("B.Font/Tests/CombatSoak/CombatSoak.lua", "rb") as f:
    raw = f.read()
    compBuf = ctypes.create_string_buffer(len(raw))
    res = OodleLZ_Compress(8, ctypes.c_char_p(raw), len(raw), compBuf, 4)
    print(f"Compressed from {len(raw)} to {res}")
"""


class Packer:
    def __init__(self, path):
        self.path = path
    
    def write_compressed(self, raw):
        while len(raw) > 0:
            raw_block_size = min(len(raw), 0x40000)
            compressed = ctypes.create_string_buffer(raw_block_size * 2)
            comp_len = OodleLZ_Compress(8, ctypes.c_char_p(raw[:raw_block_size]), raw_block_size, compressed, 4)
            if comp_len == 0:
                print("Failed to compress file")
                return
            compressed = bytes(compressed[:comp_len])
            block_size = comp_len << 2
            self.cache.write(struct.pack(">BHBI", 0x80, block_size >> 8, block_size & 0xFF, (raw_block_size << 5) | 1))
            self.cache.write(compressed)
            raw = raw[raw_block_size:]
    
    def pack(self, input_dir):
        self.toc.write(struct.pack(">Q", 0x4EC6671814000000))
        with open(os.path.join(input_dir, 'manifest.pkl'), 'rb') as f:
            manifest = pickle.load(f)
        
        for item in manifest:
            #(offset, unk1, size_comp, size_raw, unk2, parent,
            # filename) = struct.unpack("QQIIII64s", entry_bytes)
            h = ""
            if item['offset'] == 0xFFFFFFFFFFFFFFFF:
                # Directory
                self.toc.write(struct.pack("QQIIII64s", item['offset'], item['unk1'], item['size_comp'], item['size_raw'], item['unk2'], item['parent'], item['filename'].ljust(64, b'\0')))
                continue
            if item['unk1'] == 0:
                print(f"Skipping deleted entry {item['filepath']}")
                continue

            offset = self.cache.tell()
            size_raw = os.path.getsize(item['filepath'])
            if item['size_raw'] == item['size_comp']:
                # Item does not require compression
                with open(item['filepath'], 'rb') as f:
                    self.cache.write(f.read())
                    assert(self.cache.tell() - offset == size_raw)
                size_comp = size_raw
            else:
                with open(item['filepath'], 'rb') as f:
                    self.write_compressed(f.read())
                size_comp = self.cache.tell() - offset
            self.toc.write(struct.pack("QQIIII64s", offset, item['unk1'], size_comp, size_raw, item['unk2'], item['parent'], item['filename'].ljust(64, b'\0')))
    
    def __enter__(self):
        self.toc = open(self.path + '.toc', 'wb')
        self.cache = open(self.path + '.cache', 'wb')
        return self

    def __exit__(self, *args):
        self.toc.close()
        self.cache.close()


class Extractor:
    def __init__(self, path):
        self.path = path

    def extract_file(self, filepath, size_comp, size_raw, offset):
        self.cache.seek(offset)
        if size_comp == size_raw:
            return self.cache.read(size_comp)
        data = bytearray()
        while size_comp > 0:
            (header, block_size_h, block_size_l, raw_block_size) = struct.unpack(
                ">BHBI", self.cache.read(8))
            size_comp = size_comp - 8
            if (header != 0x80):
                print(
                    f"Invalid cache header for {filepath} {size_comp == size_raw}")
            flag = raw_block_size & 0x1F
            raw_block_size = raw_block_size >> 5
            if (flag != 1):
                print("Flag", flag)
            block_size = (block_size_h << 8) | block_size_l
            if block_size & 0x3:
                print("Lower block size bits are set")
            block_size = block_size >> 2
            #entries.append((filepath, size_comp, size_raw, offset, raw_block_size, block_size, offset + block_size + 8))
            comp_block = self.cache.read(block_size)

            raw_block = ctypes.create_string_buffer(raw_block_size)
            # print(filepath)
            #print("Comp size", size_comp, "Block size", block_size, "Raw size", size_raw, "Raw block size", raw_block_size)
            d_len = decompress(comp_block, block_size, raw_block,
                               raw_block_size, fuzzSafe=0, verbosity=0)

            if d_len != raw_block_size:
                print(
                    f"Failed to decompress {filepath} (expected {raw_block_size} bytes, got {d_len})")
                # print(hex(offset), size_comp, size_raw, len(dec), len(comp_buf))
                exit(0)
                return b""
            else:
                data.extend(raw_block)
                size_comp = size_comp - block_size
        if size_comp != 0:
            print(f"Unexpected compressed size: {size_comp} diff")
            exit(0)
        if len(data) != size_raw:
            print(
                f"Unexpected size after decompressing (expected {size_raw}, got {len(data)})")
            exit(0)
        return data

    def extract(self, output_dir):
        # Parse TOC
        toc_header = struct.unpack("Q", self.toc.read(8))[0]
        dirs = [output_dir + '/']
        manifest = []
        while True:
            entry_bytes = self.toc.read(96)
            if len(entry_bytes) != 96:
                break
            (offset, unk1, size_comp, size_raw, unk2, parent,
             filename) = struct.unpack("QQIIII64s", entry_bytes)
            filename = filename.rstrip(b'\x00')
            filepath = dirs[parent] + filename.decode("utf-8")
            manifest.append({'offset': offset, 'unk1': unk1, 'size_comp': size_comp,
                            'size_raw': size_raw, 'unk2': unk2, 'parent': parent, 'filename': filename, 'filepath': filepath})
            #print(hex(unk1 >> 48), hex((unk1 >> 32) & 0xFFFF), hex((unk1 >> 16) & 0xFFFF), hex((unk1) & 0xFFFF), filepath)
            if unk1 == 0:
                print("Found deleted entry ", filepath)
                continue
            # print(filepath)
            if offset == 0xFFFFFFFFFFFFFFFF:
                dirs.append(filepath + "/")
                os.makedirs(filepath, exist_ok=True)
                # entries.append((filepath, size_comp, size_raw, offset, 0))
            else:
                data = self.extract_file(filepath, size_comp, size_raw, offset)
                with open(filepath, 'wb') as f:
                    f.write(data)
        with open(output_dir + '/manifest.pkl', 'wb') as f:
            pickle.dump(manifest, f)

    def __enter__(self):
        self.toc = open(self.path + '.toc', 'rb')
        self.cache = open(self.path + '.cache', 'rb')
        return self

    def __exit__(self, *args):
        self.toc.close()
        self.cache.close()


"""
entries.sort(key=lambda x: x[2])
print(f"{'Comp': <10} {'Raw': <10} {'Offset': <10} {'RBlock': <10} {'CBlock': <10} {'Second?': <20}")
for e in entries[-20:]:
    print(e[0])
    print(f"{hex(e[1]): <10} {hex(e[2]): <10} {hex(e[3]): <10} {hex(e[4]): <10} {hex(e[5]): <10} {hex(e[6]): <10}")
"""

with Extractor("E:\\Warframe\\Downloaded\\Public\\Cache.Windows\\B.Font") as e:
    e.extract("B.Font")

#with Extractor("D:\\Darmok\\Documents\\Reversing\\wf\\original\\B.Font") as e:
    #e.extract("B.Font")

#with Extractor("B.Font") as e:
    #e.extract("B.Font_repacked")

#with Packer("B.Font") as p:
#    p.pack("B.Font")
