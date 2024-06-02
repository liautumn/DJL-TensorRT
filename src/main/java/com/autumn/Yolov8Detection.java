/*
 * 版权所有 2023 Amazon.com, Inc. 或其关联公司。保留所有权利。
 *
 * 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；除非符合许可证，否则不得使用此文件。
 * 您可以在以下网址找到许可证的副本：
 *
 * http://aws.amazon.com/apache2.0/
 *
 * 或在随附此文件的“LICENSE”文件中找到。根据许可证分发的此文件按“原样”提供，
 * 无任何明示或暗示的担保或条件。有关许可证下权限和限制的具体语言，
 * 请参阅许可证。
 */
package com.autumn;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.translator.YoloV8TranslatorFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/** 使用TensorRT引擎进行YOLOv8模型推理的示例。 */
public final class Yolov8Detection {

    private static final Logger logger = LoggerFactory.getLogger(Yolov8Detection.class);

    private Yolov8Detection() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        DetectedObjects detection = predict(); // 进行预测
        logger.info("{}", detection); // 记录检测结果
    }

    public static DetectedObjects predict() throws IOException, ModelException, TranslateException {
        // 图像路径
        Path imgPath = Paths.get("/home/images/group.jpg");
        Image img = ImageFactory.getInstance().fromFile(imgPath); // 加载图像

        // 设置Criteria，使用TensorRT引擎和.engine模型文件
        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .setTypes(Image.class, DetectedObjects.class) // 设置输入输出类型
                        .optModelPath(Paths.get("/home/model/yolov8s.engine")) // 设置模型路径
                        .optEngine("TensorRT") // 使用TensorRT引擎
                        .optArgument("width", 1024) // 输入图像宽度
                        .optArgument("height", 1024) // 输入图像高度
                        .optArgument("resize", true) // 是否调整大小
                        .optArgument("toTensor", true) // 是否转换为张量
                        .optArgument("applyRatio", true) // 是否应用比例
                        .optArgument("threshold", 0.6f) // 置信度阈值
                        .optTranslatorFactory(new YoloV8TranslatorFactory()) // 使用YOLOv8翻译器工厂
                        .optProgress(new ProgressBar()) // 进度条
                        .build();

        try (ZooModel<Image, DetectedObjects> model = criteria.loadModel();
                Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
            Path outputPath = Paths.get("/home/output"); // 输出路径
            Files.createDirectories(outputPath); // 创建输出目录

            // 进行推理并获取检测结果
            DetectedObjects detection = predictor.predict(img);
            if (detection.getNumberOfObjects() > 0) {
                img.drawBoundingBoxes(detection); // 绘制检测到的边界框
                Path output = outputPath.resolve("yolov8_detected.png"); // 输出文件路径
                try (OutputStream os = Files.newOutputStream(output)) {
                    img.save(os, "png"); // 保存标注后的图像
                }
                logger.info("检测到的对象已保存到: {}", output); // 记录输出文件路径
            }
            return detection; // 返回检测结果
        }
    }
}
