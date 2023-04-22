seg_end = [0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF]

SIGN_BIT = 0x80  # Sign bit for an A-law byte.
QUANT_MASK = 0xf  # Quantization field mask.
NSEGS = 8  # Number of A-law segments.
SEG_SHIFT = 4  # Left shift for segment number.
SEG_MASK = 0x70  # Segment field mask.

def search(val, table, size):
    for i in range(size):
        if val <= table[i]:
            return i
    return size

def linear2alaw(pcm_val):
    if pcm_val >= 0:
        mask = 0xD5  # sign (7th) bit = 1
    else:
        mask = 0x55  # sign bit = 0
        pcm_val = -pcm_val - 8

    # Convert the scaled magnitude to segment number.
    seg = search(pcm_val, seg_end, NSEGS)

    # Combine the sign, segment, and quantization bits.
    if seg >= NSEGS:  # out of range, return maximum value.
        return 0x7F ^ mask
    else:
        aval = seg << SEG_SHIFT
        if seg < 2:
            aval |= (pcm_val >> 4) & QUANT_MASK
        else:
            aval |= (pcm_val >> (seg + 3)) & QUANT_MASK
        return aval ^ mask


# The below code was converted from original code in C
# It works but we cannot use it because it undesirably increases amplitude.

#seg_uend = [0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF]
#
#def search(val, table, size):
#    for i in range(size):
#        if val <= table[i]:
#            return i
#    return size
#
#BIAS = 0x84  # Bias for linear code.
#CLIP = 8159
#
#def linear2ulaw(pcm_val):
#    # 2's complement (14-bit range)
#    mask = 0x7F if pcm_val < 0 else 0xFF
#    pcm_val = abs(pcm_val)
#    if pcm_val > CLIP:
#        pcm_val = CLIP  # clip the magnitude
#    pcm_val += BIAS >> 2
#    # Convert the scaled magnitude to segment number
#    seg = search(pcm_val, seg_uend, 8)
#    # Combine the sign, segment, quantization bits;
#    # and complement the code word.
#    if seg >= 8:  # out of range, return maximum value
#        return 0x7F ^ mask
#    else:
#        uval = (seg << 4) | ((pcm_val >> (seg + 1)) & 0xF)
#        return uval ^ mask
#


exp_lut = [0,0,1,1,2,2,2,2,3,3,3,3,3,3,3,3,
           4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
           5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
           5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
           6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
           6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
           6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
           6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
           7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
           7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
           7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
           7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
           7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
           7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
           7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
           7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]

BIAS = 0x84   # define the add-in bias for 16 bit samples
CLIP = 32635

def linear2ulaw(sample):
    # Get the sample into sign-magnitude.
    sign = (sample >> 8) & 0x80      # set aside the sign
    if sign != 0:
        sample = -sample            # get magnitude
    if sample > CLIP:
        sample = CLIP               # clip the magnitude

    # Convert from 16 bit linear to ulaw.
    sample = sample + BIAS
    exponent = exp_lut[(sample >> 7) & 0xFF]
    mantissa = (sample >> (exponent + 3)) & 0x0F
    ulawbyte = ~(sign | (exponent << 4) | mantissa)

    # optional CCITT trap
    if ulawbyte == 0:
        ulawbyte = 0x02

    return ulawbyte


