#ifndef RT_BMP_H
#define RT_BMP_H

#include <iostream>
#include "Vec3.cuh"

typedef struct SBmpHeader {
    uint16_t Header; // Decimal Value -> 19778
    uint32_t SizeOfBmpFile;
    uint16_t AppSpecific0;
    uint16_t AppSpecific1;
    uint32_t PixelArrayOffset;
} SBmpHeader;

typedef struct SDibHeader {
    uint32_t NumberOfBytes;
    uint32_t Width;
    uint32_t Height;
    uint16_t NumberOfColorPlanes;
    uint16_t BitsPerPixel;
    uint32_t UsedCompression;
    uint32_t SizeOfRawBitmap;
    uint32_t HorizontalPrintResolution;
    uint32_t VerticalPrintResolution;
    uint32_t NumberOfColorsInPalette;
    uint32_t NumberOfImportantColors;
} SDibHeader;

void SaveBmp(const char *Filepath, CVec3 *Framebuffer, int32_t FramebufferWidth, int32_t FramebufferHeight)
{
    uint8_t Padding = (4 - (FramebufferWidth * 3) % 4) % 4;

    SBmpHeader BMPheader = {
        19778U,
        uint32_t(54U + ((FramebufferWidth * 3) + Padding) * FramebufferHeight),
        0U,
        0U,
        54U
    };

    SDibHeader DIBheader = {
        40U,
        uint32_t(FramebufferWidth),
        uint32_t(FramebufferHeight),
        1U,
        sizeof(uint32_t) * 8U,
        0U,
        uint32_t(((FramebufferWidth * 3) + Padding) * FramebufferHeight),
        2835U,
        2835U,
        0U,
        0U
    };

    FILE *File;
    fopen_s(&File, Filepath, "wb");
    if (!File) {
        std::cerr << "Failed to create file!\n";
        return;
    }

    fwrite(&BMPheader.Header, sizeof(BMPheader.Header), 1, File);
    fwrite(&BMPheader.SizeOfBmpFile, sizeof(BMPheader.SizeOfBmpFile), 1, File);
    fwrite(&BMPheader.AppSpecific0, sizeof(BMPheader.AppSpecific0), 1, File);
    fwrite(&BMPheader.AppSpecific1, sizeof(BMPheader.AppSpecific1), 1, File);
    fwrite(&BMPheader.PixelArrayOffset, sizeof(BMPheader.PixelArrayOffset), 1, File);

    fwrite(&DIBheader.NumberOfBytes, sizeof(DIBheader.NumberOfBytes), 1, File);
    fwrite(&DIBheader.Width, sizeof(DIBheader.Width), 1, File);
    fwrite(&DIBheader.Height, sizeof(DIBheader.Height), 1, File);
    fwrite(&DIBheader.NumberOfColorPlanes, sizeof(DIBheader.NumberOfColorPlanes), 1, File);
    fwrite(&DIBheader.BitsPerPixel, sizeof(DIBheader.BitsPerPixel), 1, File);
    fwrite(&DIBheader.UsedCompression, sizeof(DIBheader.UsedCompression), 1, File);
    fwrite(&DIBheader.SizeOfRawBitmap, sizeof(DIBheader.SizeOfRawBitmap), 1, File);
    fwrite(&DIBheader.HorizontalPrintResolution, sizeof(DIBheader.HorizontalPrintResolution), 1, File);
    fwrite(&DIBheader.VerticalPrintResolution, sizeof(DIBheader.VerticalPrintResolution), 1, File);
    fwrite(&DIBheader.NumberOfColorsInPalette, sizeof(DIBheader.NumberOfColorsInPalette), 1, File);
    fwrite(&DIBheader.NumberOfImportantColors, sizeof(DIBheader.NumberOfImportantColors), 1, File);

    uint8_t zero = 0;
    for (int Y = 0; FramebufferHeight > Y; ++Y) {
        for (int X = 0; FramebufferWidth > X; ++X) {
            size_t PixelIndex = Y * FramebufferWidth + X;
            uint8_t IR = uint8_t(255.99 * Framebuffer[PixelIndex].R());
            uint8_t IG = uint8_t(255.99 * Framebuffer[PixelIndex].G());
            uint8_t IB = uint8_t(255.99 * Framebuffer[PixelIndex].B());
            uint8_t IA = 0xFF;
            
            uint32_t Color = (IA << 24) | (IR << 16) | (IG << 8) | IB;
            fwrite(&Color, sizeof(uint32_t), 1, File);
        }
        for (int P = 0; Padding > P; ++P) {
            fwrite(&zero, sizeof(uint8_t), 1, File);
        }
    }

    fclose(File);
}

#endif