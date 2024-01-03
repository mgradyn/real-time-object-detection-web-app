import ndarray from "ndarray";
import { Tensor } from "onnxruntime-web";
import ops from "ndarray-ops";
import ObjectDetectionCamera from "../ObjectDetectionCamera";
import { round } from "lodash";
import { yoloClasses } from "../../data/yolo_classes";
import { useState } from "react";
import { useEffect } from "react";
import { runModelUtils } from "../../utils";

const RES_TO_MODEL: [number[], string][] = [
  // [[256,256], "yolov7-tiny_256x256.onnx"],
  // [[320, 320], "yolov7-tiny_320x320.onnx"],
  // [[640, 640], "yolov7-tiny_640x640.onnx"],
  // [[640, 640], "end2end.onnx"],
  // [[640, 640], "end2end_fp16.onnx"],
  // [[256, 256], "yolov8s.ort"],
  [[256, 256], "yolov8s_opt.ort"],
  // [[640, 640], "end2end_quant.onnx"],
  // [[640, 640], "end2end.with_runtime_opt.ort"],
];

const Yolo = (props: any) => {
  const [modelResolution, setModelResolution] = useState<number[]>(
    RES_TO_MODEL[0][0]
  );
  const [modelName, setModelName] = useState<string>(RES_TO_MODEL[0][1]);
  const [session, setSession] = useState<any>(null);

  useEffect(() => {
    const getSession = async () => {
      try {
        const session = await runModelUtils.createModelCpu(
          `./_next/static/chunks/pages/${modelName}`
        );
        setSession(session);
      } catch (error) {
        console.error("Error while creating session:", error);
        // You can choose to handle the error in a specific way or rethrow it
        throw error;
      }
    };
    getSession();
  }, [modelName]);

  const changeModelResolution = () => {
    const index = RES_TO_MODEL.findIndex((item) => item[0] === modelResolution);
    if (index === RES_TO_MODEL.length - 1) {
      setModelResolution(RES_TO_MODEL[0][0]);
      setModelName(RES_TO_MODEL[0][1]);
    } else {
      setModelResolution(RES_TO_MODEL[index + 1][0]);
      setModelName(RES_TO_MODEL[index + 1][1]);
    }
  };

  const resizeCanvasCtx = (
    ctx: CanvasRenderingContext2D,
    targetWidth: number,
    targetHeight: number,
    inPlace = false
  ) => {
    let canvas: HTMLCanvasElement;

    if (inPlace) {
      // Get the canvas element that the context is associated with
      canvas = ctx.canvas;

      // Set the canvas dimensions to the target width and height
      canvas.width = targetWidth;
      canvas.height = targetHeight;

      // Scale the context to the new dimensions
      ctx.scale(
        targetWidth / canvas.clientWidth,
        targetHeight / canvas.clientHeight
      );
    } else {
      // Create a new canvas element with the target dimensions
      canvas = document.createElement("canvas");
      canvas.width = targetWidth;
      canvas.height = targetHeight;

      // Draw the source canvas into the target canvas
      canvas
        .getContext("2d")!
        .drawImage(ctx.canvas, 0, 0, targetWidth, targetHeight);

      // Get a new rendering context for the new canvas
      ctx = canvas.getContext("2d")!;
    }

    return ctx;
  };

  const preprocess = (ctx: CanvasRenderingContext2D) => {
    const resizedCtx = resizeCanvasCtx(
      ctx,
      modelResolution[0],
      modelResolution[1]
    );

    const imageData = resizedCtx.getImageData(
      0,
      0,
      modelResolution[0],
      modelResolution[1]
    );
    const { data, width, height } = imageData;
    // data processing
    const dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
    const dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [
      1,
      3,
      width,
      height,
    ]);

    ops.assign(
      dataProcessedTensor.pick(0, 0, null, null),
      dataTensor.pick(null, null, 0)
    );
    ops.assign(
      dataProcessedTensor.pick(0, 1, null, null),
      dataTensor.pick(null, null, 1)
    );
    ops.assign(
      dataProcessedTensor.pick(0, 2, null, null),
      dataTensor.pick(null, null, 2)
    );

    ops.divseq(dataProcessedTensor, 255);

    const tensor = new Tensor("float32", new Float32Array(width * height * 3), [
      1,
      3,
      width,
      height,
    ]);

    (tensor.data as Float32Array).set(dataProcessedTensor.data);
    return tensor;
  };

  const cls2color = (cls_id: number) => {
    // Assuming cld_id values are 0, 1, 2, 3, 4
    // You can customize the pastel color mapping based on your requirements
    const pastelColors = [
      "rgb(255, 100, 100)", // Higher Contrast Red
      "rgb(100, 255, 100)", // Higher Contrast Green
      "rgb(100, 100, 255)", // Higher Contrast Blue
      "rgb(255, 255, 100)", // Higher Contrast Yellow
      "rgb(255, 100, 255)", // Higher Contrast Purple
    ];

    // Ensure cld_id is within the valid range
    const index = Math.max(0, Math.min(cls_id, pastelColors.length - 1));

    return pastelColors[index];
  };

  const conf2color = (conf: number) => {
    const r = Math.round(255 * (1 - conf));
    const g = Math.round(255 * conf);
    return `rgb(${r},${g},0)`;
  };

  const postprocess = async (
    output: { dets: Tensor; labels: Tensor },
    inferenceTime: number,
    ctx: CanvasRenderingContext2D
  ): Promise<{ [key: number]: number }> => {
    // console.log(output);
    const objectCounts: { [key: number]: number } = {};

    const dx = ctx.canvas.width / modelResolution[0];
    const dy = ctx.canvas.height / modelResolution[1];

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    for (let i = 0; i < output.dets.dims[1]; i++) {
      let [x0, y0, x1, y1, score] = output.dets.data.slice(i * 5, i * 5 + 5);

      const decimalScore = parseFloat(score as string);

      // Check if the decimalScore is less than 0.5
      if (decimalScore < 0.5) {
        continue;
      }

      const cls_id = Number(output.labels.data[i]);

      // scale to canvas size
      [x0, x1] = [x0, x1].map((x: any) => x * dx);
      [y0, y1] = [y0, y1].map((x: any) => x * dy);

      [x0, y0, x1, y1] = [x0, y0, x1, y1].map((x: any) => round(x));

      [score] = [score].map((x: any) => round(x * 100, 1));
      const label =
        yoloClasses[cls_id].toString()[0].toUpperCase() +
        yoloClasses[cls_id].toString().substring(1) +
        " " +
        score.toString() +
        "%";
      // const color = conf2color(score / 100);

      objectCounts[cls_id] = (objectCounts[cls_id] || 0) + 1;

      const color = cls2color(cls_id);

      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.font = "20px Arial";
      ctx.fillStyle = color;
      ctx.fillText(label, x0, y0 - 5);

      // fillrect with transparent color
      ctx.fillStyle = color.replace(")", ", 0.1)").replace("rgb", "rgba");
      ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
    }
    return objectCounts;
  };

  return (
    <ObjectDetectionCamera
      width={props.width}
      height={props.height}
      preprocess={preprocess}
      postprocess={postprocess}
      resizeCanvasCtx={resizeCanvasCtx}
      session={session}
      changeModelResolution={changeModelResolution}
      modelName={modelName}
    />
  );
};

export default Yolo;
