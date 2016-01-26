#ifndef PARSE_SIZE_T_H
#define PARSE_SIZE_T_H

size_t parse_size_t(const char *str)
// Parse 100K, 1G etc. Return 0 for error.
{
    size_t n;
    char c;
    if (sscanf(str, "%zu%c", &n, &c) == 2) // number with common suffix
        switch (c) {
            case 'K': case 'k': // 100K, 200K, etc
                return n*1000;
            case 'M': case 'm': // 100M, 200M, etc
                return n*1000000;
            case 'B': case 'b': // 1B, 2B, etc
                return n*1000000000;
            default:
                return 0; // error
        }
    if (sscanf(str, "%zu", &n) == 1) // plain number
        return n;
    return 0; // error
}

#endif
