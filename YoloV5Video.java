import Utils.ImageGUI;
import Utils.ImageUtil;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.YoloV5Translator;
//import ai.djl.modality.cv.translator.YoloV5TranslatorFactory;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import org.bytedeco.javacv.CanvasFrame;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.video.Video;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;
import org.opencv.videoio.Videoio;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.Date;
import java.util.List;

public class YoloV5Video {
    private static Predictor<Image, DetectedObjects> yoloV5Predictor;

    public YoloV5Video() throws TranslateException, ModelNotFoundException, MalformedModelException, IOException {
        getYoloV5Predictor();
        System.out.println("1");
    }

    public static void main(String[] args) throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {
        URL url = ClassLoader.getSystemResource("lib/opencv_java455.dll");
        System.load(url.getPath());
        new YoloV5Video();

        NDManager Nd = NDManager.newBaseManager();


        Video video = new Video();
        VideoCapture capture = new VideoCapture();

        capture.open(0);
        if(!capture.isOpened()) {
            System.out.println("could not load video data...");
            return;
        }
        int frame_width = (int)capture.get(3);
        int frame_height = (int)capture.get(4);
//        ImageGUI gui = new ImageGUI();
//        gui.createWin("YoloV5", new Dimension(frame_width, frame_height));
        CanvasFrame canvasFrame = new CanvasFrame("123");
        canvasFrame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        canvasFrame.setAlwaysOnTop(true);
        Mat frame = new Mat();
        Size size = new Size(frame_width, frame_height);
        VideoWriter videoWriter = new VideoWriter("./data/files/video/2.mp4",VideoWriter.fourcc('D','I','V','X'),30,size,true);

        int fps = 0;
        while(true) {
            boolean have = capture.read(frame);
            Core.flip(frame, frame, 1);// Win上摄像头
            if(!have) break;
            if(!frame.empty()) {
                BufferedImage bufferedImage = conver2Image(frame);
                Image image = ImageFactory.getInstance().fromImage(bufferedImage);
                DetectedObjects predict = yoloV5Predictor.predict(image);
                ImageUtil.drawBoundingBoxes(bufferedImage,predict);
                fps++;
                Mat mat = convertMat(bufferedImage);
                videoWriter.write(mat);
//                System.out.println(videoWriter.isOpened());
//                gui.imshow(bufferedImage);
//                gui.repaint();
                canvasFrame.showImage(bufferedImage);
                if (fps/30 > 20){
                    capture.release();
                    videoWriter.release();
                    System.exit(0);
                    break;
                }


            }
//            try {
//                Thread.sleep(100);
//            } catch (InterruptedException e) {
//                e.printStackTrace();
//            }
        }

    }

    public void getYoloV5Predictor() throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {
        //输入图片的大小 对应导出torchscript的大小
        int imageSize = 640;
        List<String> labels = Arrays.asList("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                "hair drier", "toothbrush");
        //图片处理步骤
        Pipeline pipeline = new Pipeline();
        pipeline.add(new Resize(imageSize)); //调整尺寸
        pipeline.add(new ToTensor()); //处理为tensor类型
        //定义YoLov5的Translator
        Translator<ai.djl.modality.cv.Image, DetectedObjects> translator =  YoloV5Translator
                .builder()
                .setPipeline(pipeline)
                //labels信息定义
//                .optSynsetArtifactName("coco.names") //数据的labels文件名称
                .optSynset(labels) //数据的labels数据
                //预测的最小下限
                .optThreshold(0.5f)
                .optOutputType(YoloV5Translator.YoloOutputType.AUTO)
                .build();
        //构建Model Criteria
        Criteria<ai.djl.modality.cv.Image, DetectedObjects> criteria = Criteria.builder()
                .setTypes(ai.djl.modality.cv.Image.class, DetectedObjects.class)//图片目标检测类型
                .optModelUrls("data/yolov5")//模型的路径
                .optModelName("yolov5m.torchscript.pt")//模型的文件名称
                .optDevice(Device.cpu())

                .optTranslator(translator)//设置Translator
                .optProgress(new ProgressBar())//展示加载进度
                .build();
        //加载Model
        ZooModel<Image,DetectedObjects> model = ModelZoo.loadModel(criteria);
        Predictor<Image, DetectedObjects> yoloV5Predictor = model.newPredictor();
        this.yoloV5Predictor = yoloV5Predictor;



    }
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

