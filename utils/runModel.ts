import { InferenceSession, Tensor } from "onnxruntime-web";

export async function createModelCpu(url: string): Promise<InferenceSession> {
  return await InferenceSession.create(url, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
}

export async function runModel(
  model: InferenceSession,
  preprocessedData: Tensor
): Promise<[{ dets: Tensor; labels: Tensor }, number]> {
  try {
    const feeds: Record<string, Tensor> = {};
    feeds[model.inputNames[0]] = preprocessedData;
    const start = Date.now();
    const outputData = await model.run(feeds);

    // console.log(outputData)

    const end = Date.now();
    const inferenceTime = end - start;
    const dets = outputData[model.outputNames[0]];
    const labels = outputData[model.outputNames[1]];

    const output = { dets, labels };

    return [output, inferenceTime];
  } catch (e) {
    console.error(e);
    throw new Error();
  }
}
