#ifndef BUFFER_H
#define BUFFER_H

template <typename T>
class Buffer {
public:
    size_t length;
    T *buf;
    size_t cursor;

    Buffer(void)
        : length(0), buf(nullptr), cursor(0) {}

    Buffer(size_t length, T *buf, size_t cursor)
        : length(length), buf(buf), cursor(cursor) {}

    void set(T elem)
    {
        buf[cursor] = elem;
    }

    T get(void) const
    {
        return buf[cursor];
    }
};

template <typename T>
size_t fetch(const Buffer<T> &dest, const Buffer<T> &src, size_t nelem)
// fetches at most nelem
{
    if (src.cursor + nelem > src.length)
        nelem = src.length - src.cursor;
    if (dest.cursor + nelem > dest.length)
        nelem = dest.length - dest.cursor;

    memcpy(&dest.buf[dest.cursor], &src.buf[src.cursor], nelem*sizeof(T));

    return nelem;
}

template <typename T>
Buffer<T> fetch_and_slice(const Buffer<T> &dest,
                          const Buffer<T> &src, size_t nelem)
{
    nelem = fetch(dest, src, nelem);

    return Buffer<T>(nelem, &dest.buf[dest.cursor], 0);
}

template <typename T>
size_t fetch_into_gpu(const Buffer<T> &dest,
                      const Buffer<T> &src, size_t nelem)
{
    if (src.cursor + nelem > src.length)
        nelem = src.length - src.cursor;
    if (dest.cursor + nelem > dest.length)
        nelem = dest.length - dest.cursor;

    cudaMemcpy(&dest.buf[dest.cursor], &src.buf[src.cursor],
        nelem*sizeof(T), cudaMemcpyHostToDevice);

    return nelem;
}

template <typename T>
Buffer<T> fetch_into_gpu_and_slice(const Buffer<T> &dest,
                                   const Buffer<T> &src, size_t nelem)
{
    nelem = fetch_into_gpu(dest, src, nelem);

    return Buffer<T>(nelem, &dest.buf[dest.cursor], 0);
}

template <typename T>
size_t fetch_into_gpu_async(const Buffer<T> &dest, const Buffer<T> &src,
    size_t nelem, cudaStream_t stream)
{
    if (src.cursor + nelem > src.length)
        nelem = src.length - src.cursor;
    if (dest.cursor + nelem > dest.length)
        nelem = dest.length - dest.cursor;

    cudaMemcpyAsync(&dest.buf[dest.cursor], &src.buf[src.cursor],
        nelem*sizeof(T), cudaMemcpyHostToDevice, stream);

    return nelem;
}

template <typename T>
Buffer<T> fetch_into_gpu_async_and_slice(const Buffer<T> &dest,
    const Buffer<T> &src, size_t nelem, cudaStream_t stream)
{
    nelem = fetch_into_gpu_async(dest, src, nelem, stream);

    return Buffer<T>(nelem, &dest.buf[dest.cursor], 0);
}

template <typename T>
size_t fetch_from_gpu(const Buffer<T> &dest,
                      const Buffer<T> &src, size_t nelem)
{
    if (src.cursor + nelem > src.length)
        nelem = src.length - src.cursor;
    if (dest.cursor + nelem > dest.length)
        nelem = dest.length - dest.cursor;

    cudaMemcpy(&dest.buf[dest.cursor], &src.buf[src.cursor],
        nelem*sizeof(T), cudaMemcpyDeviceToHost);

    return nelem;
}

template <typename T>
Buffer<T> fetch_from_gpu_and_slice(const Buffer<T> &dest,
                                   const Buffer<T> &src, size_t nelem)
{
    nelem = fetch_from_gpu(dest, src, nelem);

    return Buffer<T>(nelem, &dest.buf[dest.cursor], 0);
}

template <typename T>
size_t fetch_from_gpu_async(const Buffer<T> &dest,
    const Buffer<T> &src, size_t nelem, cudaStream_t stream)
{
    if (src.cursor + nelem > src.length)
        nelem = src.length - src.cursor;
    if (dest.cursor + nelem > dest.length)
        nelem = dest.length - dest.cursor;

    cudaMemcpyAsync(&dest.buf[dest.cursor], &src.buf[src.cursor],
        nelem*sizeof(T), cudaMemcpyDeviceToHost, stream);

    return nelem;
}

template <typename T>
Buffer<T> fetch_from_gpu_async_and_slice(const Buffer<T> &dest,
    const Buffer<T> &src, size_t nelem, cudaStream_t stream)
{
    nelem = fetch_from_gpu_async(dest, src, nelem, stream);

    return Buffer<T>(nelem, &dest.buf[dest.cursor], 0);
}

#endif
