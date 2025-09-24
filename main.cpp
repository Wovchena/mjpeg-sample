#include <cstdint>
#include <vector>
#include <random>
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <stdexcept>

// Bring in stb_image_write for JPEG encoding to memory.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ------------------------------
// Small helpers for little-endian I/O
// ------------------------------
struct LEWriter {
    std::ofstream out;

    explicit LEWriter(const std::string& path) {
        out.open(path, std::ios::binary);
        if (!out) throw std::runtime_error("Failed to open output file: " + path);
    }
    uint32_t tell() { return static_cast<uint32_t>(out.tellp()); }

    void seek(uint32_t pos) {
        out.seekp(pos, std::ios::beg);
        if (!out) throw std::runtime_error("seek failed");
    }

    void write_bytes(const void* data, size_t n) {
        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
        if (!out) throw std::runtime_error("write failed");
    }

    void write_u16(uint16_t v) {
        uint8_t b[2] = {
            static_cast<uint8_t>( (v >> 0) & 0xFF ),
            static_cast<uint8_t>( (v >> 8) & 0xFF )
        };
        write_bytes(b, 2);
    }
    void write_u32(uint32_t v) {
        uint8_t b[4] = {
            static_cast<uint8_t>( (v >> 0) & 0xFF ),
            static_cast<uint8_t>( (v >> 8) & 0xFF ),
            static_cast<uint8_t>( (v >> 16) & 0xFF ),
            static_cast<uint8_t>( (v >> 24) & 0xFF )
        };
        write_bytes(b, 4);
    }
    void write_i32(int32_t v) { write_u32(static_cast<uint32_t>(v)); }

    void write_fourcc(const char tag[4]) { write_bytes(tag, 4); }

    void patch_u32(uint32_t pos, uint32_t v) {
        auto cur = tell();
        seek(pos);
        write_u32(v);
        seek(cur);
    }
};

// ------------------------------
// AVI MJPEG writer
// ------------------------------
struct AVIIndexEntry {
    uint32_t offset; // relative to start of 'movi' data (after the "movi" FOURCC)
    uint32_t size;   // data size (excluding padding)
};

class AVIMJPEGWriter {
public:
    AVIMJPEGWriter(const std::string& path, uint32_t width, uint32_t height, uint32_t fps)
        : w(width), h(height), fps(fps), writer(path)
    {
        begin_file();
    }

    ~AVIMJPEGWriter() {
        try {
            if (!finalized) finish();
        } catch (...) {
            // avoid throwing in dtor
        }
    }

    // Adds a single JPEG-compressed frame as an MJPEG chunk.
    // 'rgb' must be w*h*3 (RGB24).
    void add_rgb_frame_as_jpeg(const uint8_t* rgb, int quality = 85) {
        // Encode to JPEG in-memory using stb (no temp files).
        std::vector<uint8_t> jpeg;
        jpeg.reserve(w * h / 2); // heuristic
        stbi_write_jpg_to_func(&stb_callback, &jpeg, static_cast<int>(w), static_cast<int>(h),
                               3, rgb, quality);

        // Write '00dc' chunk into 'movi' list
        const uint32_t chunk_start = writer.tell();
        writer.write_fourcc("00dc");
        writer.write_u32(static_cast<uint32_t>(jpeg.size())); // size excluding padding
        writer.write_bytes(jpeg.data(), jpeg.size());
        if (jpeg.size() & 1) {
            const uint8_t pad = 0;
            writer.write_bytes(&pad, 1); // word-align
        }

        // Record for idx1
        AVIIndexEntry e;
        e.offset = chunk_start - movi_data_start; // offset to the chunk header within movi
        e.size   = static_cast<uint32_t>(jpeg.size());
        index.push_back(e);

        ++frame_count;
        if (jpeg.size() > max_chunk) max_chunk = static_cast<uint32_t>(jpeg.size());
    }

    void finish() {
        if (finalized) return;

        // Close 'movi' list: compute and patch its size
        const uint32_t movi_end = writer.tell();
        const uint32_t movi_size = movi_end - (movi_list_start + 8); // size excludes 'LIST' + size field
        writer.patch_u32(movi_list_size_pos, movi_size);

        // Write 'idx1'
        writer.write_fourcc("idx1");
        writer.write_u32(static_cast<uint32_t>(index.size()) * 16);
        for (const auto& e : index) {
            writer.write_fourcc("00dc");       // dwChunkId
            writer.write_u32(0x00000010);      // dwFlags (AVIIF_KEYFRAME)
            writer.write_u32(e.offset);        // dwOffset (from movi data start)
            writer.write_u32(e.size);          // dwSize (data only)
        }

        // Patch header fields now that we know counts/sizes
        const uint32_t file_end = writer.tell();

        // Patch 'hdrl' list size
        const uint32_t hdrl_size = hdrl_end - (hdrl_list_start + 8);
        writer.patch_u32(hdrl_list_size_pos, hdrl_size);

        // Patch 'RIFF' size
        const uint32_t riff_size = file_end - (riff_start + 8);
        writer.patch_u32(riff_size_pos, riff_size);

        // Patch avih: total frames, suggested buffer, max bytes/sec
        writer.patch_u32(avih_data_pos + 16, frame_count);                // dwTotalFrames
        writer.patch_u32(avih_data_pos + 28, max_chunk);                  // dwSuggestedBufferSize
        writer.patch_u32(avih_data_pos +  4, fps * max_chunk);            // dwMaxBytesPerSec

        // Patch strh: length (in frames) + suggested buffer size
        writer.patch_u32(strh_data_pos + 32, frame_count);                // dwLength
        writer.patch_u32(strh_data_pos + 36, max_chunk);                  // dwSuggestedBufferSize

        finalized = true;
    }

    uint32_t width() const { return w; }
    uint32_t height() const { return h; }
    uint32_t frame_rate() const { return fps; }

private:
    uint32_t w, h, fps;

    LEWriter writer;

    // Offsets for patching
    uint32_t riff_start = 0;
    uint32_t riff_size_pos = 0;

    uint32_t hdrl_list_start = 0;
    uint32_t hdrl_list_size_pos = 0;
    uint32_t hdrl_end = 0;

    uint32_t avih_data_pos = 0;
    uint32_t strh_data_pos = 0;

    uint32_t movi_list_start = 0;
    uint32_t movi_list_size_pos = 0;
    uint32_t movi_data_start = 0;

    std::vector<AVIIndexEntry> index;
    uint32_t frame_count = 0;
    uint32_t max_chunk = 0;
    bool finalized = false;

    static void stb_callback(void* ctx, void* data, int size) {
        auto* vec = reinterpret_cast<std::vector<uint8_t>*>(ctx);
        auto* p   = reinterpret_cast<const uint8_t*>(data);
        vec->insert(vec->end(), p, p + size);
    }

    void begin_file() {
        // RIFF header
        riff_start = writer.tell();
        writer.write_fourcc("RIFF");
        riff_size_pos = writer.tell();
        writer.write_u32(0); // placeholder for RIFF size
        writer.write_fourcc("AVI ");

        // LIST 'hdrl'
        hdrl_list_start = writer.tell();
        writer.write_fourcc("LIST");
        hdrl_list_size_pos = writer.tell();
        writer.write_u32(0); // placeholder
        writer.write_fourcc("hdrl");

        // 'avih' (MainAVIHeader) 56 bytes
        writer.write_fourcc("avih");
        writer.write_u32(56);  // chunk size
        avih_data_pos = writer.tell();

        const uint32_t usec_per_frame = 1000000u / fps;
        writer.write_u32(usec_per_frame); // dwMicroSecPerFrame
        writer.write_u32(0);              // dwMaxBytesPerSec (patch later)
        writer.write_u32(0);              // dwPaddingGranularity
        writer.write_u32(0x00000010);     // dwFlags (AVIF_HASINDEX)
        writer.write_u32(0);              // dwTotalFrames (patch later)
        writer.write_u32(0);              // dwInitialFrames
        writer.write_u32(1);              // dwStreams
        writer.write_u32(0);              // dwSuggestedBufferSize (patch later)
        writer.write_u32(w);              // dwWidth
        writer.write_u32(h);              // dwHeight
        writer.write_u32(0); writer.write_u32(0); writer.write_u32(0); writer.write_u32(0); // dwReserved[4]

        // LIST 'strl'
        const uint32_t strl_list_start = writer.tell();
        writer.write_fourcc("LIST");
        const uint32_t strl_list_size_pos = writer.tell();
        writer.write_u32(0); // placeholder
        writer.write_fourcc("strl");

        // 'strh' (AVISTREAMHEADER) 56 bytes
        writer.write_fourcc("strh");
        writer.write_u32(56); // chunk size
        strh_data_pos = writer.tell();

        writer.write_fourcc("vids");      // fccType
        writer.write_fourcc("MJPG");      // fccHandler
        writer.write_u32(0);              // dwFlags
        writer.write_u16(0);              // wPriority
        writer.write_u16(0);              // wLanguage
        writer.write_u32(0);              // dwInitialFrames
        writer.write_u32(1);              // dwScale
        writer.write_u32(fps);            // dwRate
        writer.write_u32(0);              // dwStart
        writer.write_u32(0);              // dwLength (patch later)
        writer.write_u32(0);              // dwSuggestedBufferSize (patch later)
        writer.write_u32(0xFFFFFFFFu);    // dwQuality (default)
        writer.write_u32(0);              // dwSampleSize (0 = variable)

        // rcFrame (4 * 16-bit) -> left, top, right, bottom
        writer.write_u16(0); // left
        writer.write_u16(0); // top
        writer.write_u16(static_cast<uint16_t>(w)); // right
        writer.write_u16(static_cast<uint16_t>(h)); // bottom

        // 'strf' (BITMAPINFOHEADER) 40 bytes
        writer.write_fourcc("strf");
        writer.write_u32(40);             // chunk size
        writer.write_u32(40);             // biSize
        writer.write_i32(static_cast<int32_t>(w)); // biWidth
        writer.write_i32(static_cast<int32_t>(h)); // biHeight
        writer.write_u16(1);              // biPlanes
        writer.write_u16(24);             // biBitCount (RGB24 before compression)
        writer.write_fourcc("MJPG");      // biCompression
        writer.write_u32(0);              // biSizeImage (can be 0 for MJPG)
        writer.write_i32(0);              // biXPelsPerMeter
        writer.write_i32(0);              // biYPelsPerMeter
        writer.write_u32(0);              // biClrUsed
        writer.write_u32(0);              // biClrImportant

        // Patch strl LIST size
        const uint32_t strl_end = writer.tell();
        const uint32_t strl_size = strl_end - (strl_list_start + 8);
        writer.patch_u32(strl_list_size_pos, strl_size);

        // hdrl end for later size computation
        hdrl_end = writer.tell();

        // We'll patch hdrl list size at finish()

        // LIST 'movi'
        movi_list_start = writer.tell();
        writer.write_fourcc("LIST");
        movi_list_size_pos = writer.tell();
        writer.write_u32(0); // placeholder
        writer.write_fourcc("movi");
        movi_data_start = writer.tell(); // after the "movi" FOURCC
    }
};

// ------------------------------
// Random frame generator
// ------------------------------
static void fill_random_rgb(std::vector<uint8_t>& buf, uint32_t w, uint32_t h) {
    std::mt19937 rng(1337);
    std::uniform_int_distribution<int> dist(0, 255);
    const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h) * 3;
    buf.resize(n);
    for (size_t i = 0; i < n; ++i) {
        buf[i] = static_cast<uint8_t>(dist(rng));
    }
}

int main(int argc, char** argv) {
    try {
        // Defaults
        std::string out_path = "random_mjpeg.avi";
        uint32_t width = 640;
        uint32_t height = 360;
        uint32_t fps = 30;
        uint32_t seconds = 5;
        int quality = 85;

        // Simple CLI override: program [out.avi] [width] [height] [fps] [seconds] [quality]
        if (argc >= 2) out_path = argv[1];
        if (argc >= 3) width = static_cast<uint32_t>(std::stoul(argv[2]));
        if (argc >= 4) height = static_cast<uint32_t>(std::stoul(argv[3]));
        if (argc >= 5) fps = static_cast<uint32_t>(std::stoul(argv[4]));
        if (argc >= 6) seconds = static_cast<uint32_t>(std::stoul(argv[5]));
        if (argc >= 7) quality = std::stoi(argv[6]);

        AVIMJPEGWriter avi(out_path, width, height, fps);

        const uint32_t total_frames = fps * seconds;

        // We'll re-use the buffer and re-randomize it each frame
        std::vector<uint8_t> rgb;
        rgb.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, 255);

        for (uint32_t i = 0; i < total_frames; ++i) {
            for (auto& px : rgb) {
                px = static_cast<uint8_t>(dist(rng));
            }
            avi.add_rgb_frame_as_jpeg(rgb.data(), quality);
        }

        avi.finish();

        std::cout << "Wrote " << out_path << " (" << total_frames
                  << " frames, " << width << "x" << height << " @ " << fps << " fps)\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }
}
