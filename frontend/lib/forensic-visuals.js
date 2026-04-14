function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Failed to decode image."));
    image.src = src;
  });
}

function makeCanvas(width, height, smoothingEnabled = true) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = smoothingEnabled;
  return { canvas, ctx };
}

export async function buildComparisonViews(
  previewDataUrl,
  maskDataUrl,
  overlayDataUrl = "",
  overlayAlpha = 0.4
) {
  if (!previewDataUrl) {
    return {
      originalSrc: "",
      detectedRegionSrc: "",
      overlaySrc: overlayDataUrl,
      hasMask: false,
    };
  }

  if (!maskDataUrl) {
    return {
      originalSrc: previewDataUrl,
      detectedRegionSrc: "",
      overlaySrc: overlayDataUrl,
      hasMask: false,
    };
  }

  const [sourceImage, maskImage] = await Promise.all([
    loadImage(previewDataUrl),
    loadImage(maskDataUrl),
  ]);

  const width = sourceImage.naturalWidth || sourceImage.width;
  const height = sourceImage.naturalHeight || sourceImage.height;

  const { canvas: sourceCanvas, ctx: sourceCtx } = makeCanvas(width, height);
  sourceCtx.drawImage(sourceImage, 0, 0, width, height);

  const { canvas: scaledMaskCanvas, ctx: scaledMaskCtx } = makeCanvas(
    width,
    height,
    false
  );
  scaledMaskCtx.drawImage(maskImage, 0, 0, width, height);

  const maskPixels = scaledMaskCtx.getImageData(0, 0, width, height).data;
  let hasMask = false;
  for (let index = 0; index < maskPixels.length; index += 4) {
    if (maskPixels[index] > 0) {
      hasMask = true;
      break;
    }
  }

  const { canvas: isolatedCanvas, ctx: isolatedCtx } = makeCanvas(width, height);
  isolatedCtx.drawImage(sourceCanvas, 0, 0);
  isolatedCtx.globalCompositeOperation = "destination-in";
  isolatedCtx.drawImage(scaledMaskCanvas, 0, 0);
  isolatedCtx.globalCompositeOperation = "source-over";

  const { canvas: blackCanvas, ctx: blackCtx } = makeCanvas(width, height);
  blackCtx.fillStyle = "#000000";
  blackCtx.fillRect(0, 0, width, height);
  if (hasMask) {
    blackCtx.drawImage(isolatedCanvas, 0, 0);
  }

  const { canvas: tintedMaskCanvas, ctx: tintedMaskCtx } = makeCanvas(
    width,
    height
  );
  tintedMaskCtx.fillStyle = "rgb(255, 59, 48)";
  tintedMaskCtx.fillRect(0, 0, width, height);
  tintedMaskCtx.globalCompositeOperation = "destination-in";
  tintedMaskCtx.drawImage(scaledMaskCanvas, 0, 0);
  tintedMaskCtx.globalCompositeOperation = "source-over";

  const { canvas: overlayCanvas, ctx: overlayCtx } = makeCanvas(width, height);
  if (hasMask) {
    overlayCtx.drawImage(sourceCanvas, 0, 0);
    overlayCtx.globalAlpha = overlayAlpha;
    overlayCtx.drawImage(tintedMaskCanvas, 0, 0);
    overlayCtx.globalAlpha = 1;
  }

  return {
    originalSrc: previewDataUrl,
    detectedRegionSrc: blackCanvas.toDataURL("image/png"),
    overlaySrc: hasMask
      ? overlayDataUrl || overlayCanvas.toDataURL("image/png")
      : "",
    hasMask,
  };
}
