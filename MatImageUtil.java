package Utils;

import org.bytedeco.opencv.opencv_core.IplImage;
import org.opencv.core.Mat;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

public class MatImageUtil {
    public static BufferedImage conver2Image(Mat mat) {
        int width = mat.cols();
        int height = mat.rows();
        int dims = mat.channels();
        int[] pixels = new int[width*height];
        byte[] rgbdata = new byte[width*height*dims];
        mat.get(0, 0, rgbdata);
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        int index = 0;
        int r=0, g=0, b=0;
        for(int row=0; row<height; row++) {
            for(int col=0; col<width; col++) {
                if(dims == 3) {
                    index = row*width*dims + col*dims;
                    b = rgbdata[index]&0xff;
                    g = rgbdata[index+1]&0xff;
                    r = rgbdata[index+2]&0xff;
                    pixels[row*width+col] = ((255&0xff)<<24) | ((r&0xff)<<16) | ((g&0xff)<<8) | b&0xff;
                }
                if(dims == 1) {
                    index = row*width + col;
                    b = rgbdata[index]&0xff;
                    pixels[row*width+col] = ((255&0xff)<<24) | ((b&0xff)<<16) | ((b&0xff)<<8) | b&0xff;
                }
            }
        }
        setRGB( image, 0, 0, width, height, pixels);
        return image;
    }

//    public static BufferedImage iplToBufferedImage(IplImage iplImage) {
//        if (iplImage.height() > 0 && iplImage.width() > 0) {
//            BufferedImage image = new BufferedImage(iplImage.width(), iplImage.height(), BufferedImage.TYPE_3BYTE_BGR);
//            iplImage.
//            ;
//            return image;
//        }
//        return null;
//
//    }
    public static Mat convertMat(BufferedImage im) {
        // Convert INT to BYTE
        im = toBufferedImageOfType(im, BufferedImage.TYPE_3BYTE_BGR);
        // Convert bufferedimage to byte array
        byte[] pixels = ((DataBufferByte) im.getRaster().getDataBuffer())
                .getData();
        // Create a Matrix the same size of image
        Mat image = new Mat(im.getHeight(), im.getWidth(), 16);
        // Fill Matrix with image values
        image.put(0, 0, pixels);
        return image;
    }


    private static BufferedImage toBufferedImageOfType(BufferedImage original, int type) {
        if (original == null) {
            throw new IllegalArgumentException("original == null");
        }

        // Don't convert if it already has correct type
        if (original.getType() == type) {
            return original;
        }
        // Create a buffered image
        BufferedImage image = new BufferedImage(original.getWidth(), original.getHeight(), type);
        // Draw the image onto the new buffer
        Graphics2D g = image.createGraphics();
        try {
            g.setComposite(AlphaComposite.Src);
            g.drawImage(original, 0, 0, null);
        } finally {
            g.dispose();
        }

        return image;
    }

    public static void setRGB( BufferedImage image, int x, int y, int width, int height, int[] pixels ) {
        int type = image.getType();
        if ( type == BufferedImage.TYPE_INT_ARGB || type == BufferedImage.TYPE_INT_RGB )
            image.getRaster().setDataElements( x, y, width, height, pixels );
        else
            image.setRGB( x, y, width, height, pixels, 0, width );
    }
}
