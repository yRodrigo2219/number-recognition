import Network from "./Network.js";
import NetworkState from "./data/network.json";
import TestInput from "./data/testInput.json";

window.onload = async function () {
  const guessElement = document.getElementById("guess");
  const border_div = document.getElementById("border");

  const canvas_ui = document.getElementById("canvas_ui");
  const canvas_gray = document.createElement("canvas");
  const canvas_nn = document.getElementById("canvas_nn");

  canvas_gray.width = canvas_ui.width;
  canvas_gray.height = canvas_ui.height;

  const ctx_ui = canvas_ui.getContext("2d");
  const ctx_gray = canvas_gray.getContext("2d");
  const ctx_nn = canvas_nn.getContext("2d");

  ctx_nn.imageSmoothingEnabled = true;

  const lineWidth = 80;
  const box = {
    top: canvas_ui.height / 2,
    left: canvas_ui.width / 2,
    bottom: 0,
    right: 0,
  };
  const coords = { x: 0, y: 0 };
  let painting = false;

  const data = NetworkState;
  const net = new Network(data);
  guessElement.innerText = getResult(net, TestInput);

  //ctx_gray.filter = "grayscale(1)";
  clearCanvas();

  /* IMAGE TEST
   */
  const imageData = ctx_nn.getImageData(
    0,
    0,
    canvas_nn.width,
    canvas_nn.height
  );
  const idata = imageData.data;
  console.log(TestInput);
  for (let i = 0, j = 0; i < idata.length; i += 4, j++) {
    idata[i] = (1 - TestInput[j][0]) * 255;
    idata[i + 1] = (1 - TestInput[j][0]) * 255;
    idata[i + 2] = (1 - TestInput[j][0]) * 255;
    idata[i + 3] = 255;
  }

  ctx_nn.putImageData(imageData, 0, 0);

  canvas_ui.addEventListener("mousedown", startDrawing);
  canvas_ui.addEventListener("mouseup", stopDrawing);
  canvas_ui.addEventListener("mousemove", draw);

  function startDrawing(event) {
    clearCanvas();
    painting = true;
    getPosition(event);
  }

  function stopDrawing() {
    painting = false;
    canvasToImage();
    resetBox();
  }

  function draw(event) {
    if (!painting) return;

    ctx_ui.beginPath();
    ctx_gray.beginPath();
    ctx_ui.lineWidth = lineWidth;
    ctx_gray.lineWidth = lineWidth;
    ctx_ui.lineCap = "round";
    ctx_gray.lineCap = "square";
    ctx_ui.strokeStyle = "#FFC75F";
    ctx_gray.strokeStyle = "black";
    ctx_ui.moveTo(coords.x, coords.y);
    ctx_gray.moveTo(coords.x, coords.y);
    getPosition(event);
    ctx_ui.lineTo(coords.x, coords.y);
    ctx_gray.lineTo(coords.x, coords.y);
    ctx_ui.stroke();
    ctx_gray.stroke();
  }

  function getPosition(event) {
    const rect = canvas_ui.getBoundingClientRect(),
      scaleX = canvas_ui.width / rect.width,
      scaleY = canvas_ui.height / rect.height;

    coords.x = (event.clientX - rect.left) * scaleX;
    coords.y = (event.clientY - rect.top) * scaleY;

    const top = coords.y - lineWidth / 2;
    const bottom = coords.y + lineWidth / 2;
    const left = coords.x - lineWidth / 2;
    const right = coords.x + lineWidth / 2;

    //update box
    if (box.top > top) box.top = top;
    if (box.left > left) box.left = left;
    if (box.bottom < bottom) box.bottom = bottom;
    if (box.right < right) box.right = right;

    showBox();
  }

  function canvasToImage() {
    const { sTop, sLeft, sWidth, sHeight } = {
      sTop: box.top,
      sLeft: box.left,
      sWidth: box.right - box.left,
      sHeight: box.bottom - box.top,
    };

    const dWidth = sWidth / sHeight >= 1 ? 20 : 20 * (sWidth / sHeight);
    const dHeight = sWidth / sHeight < 1 ? 20 : 20 * (sWidth / sHeight);
    const dTop = (28 - dHeight) / 2;
    const dLeft = (28 - dWidth) / 2;

    ctx_nn.drawImage(
      canvas_gray,
      sLeft,
      sTop,
      sWidth,
      sHeight,
      dLeft,
      dTop,
      dWidth,
      dHeight
    );

    const imageData = ctx_nn.getImageData(
      0,
      0,
      canvas_nn.width,
      canvas_nn.height
    ).data;

    const imageMatrix = [];
    for (let i = 0, j = 0; i < imageData.length; i += 4, j++) {
      const gray =
        1 -
        ((imageData[i] + imageData[i + 1] + imageData[i + 2]) *
          (imageData[i + 3] / 255)) /
          765; // 0...1

      imageMatrix[j] = [parseFloat(gray)];
    }

    guessElement.innerText = getResult(net, imageMatrix);
  }

  function showBox() {
    const smaller = 2;

    border_div.style.cssText = `
      display: initial;
      left: ${box.left / smaller}px;
      top: ${(box.top + canvas_ui.offsetTop * 2) / smaller}px;
      width: ${(box.right - box.left) / smaller}px;
      height: ${
        (box.bottom +
          canvas_ui.offsetTop * 2 -
          (box.top + canvas_ui.offsetTop * 2)) /
        smaller
      }px;
      border: 1px solid blue;
    `;
  }

  function resetBox() {
    box.top = canvas_ui.height / 2;
    box.left = canvas_ui.width / 2;
    box.bottom = 0;
    box.right = 0;
    border_div.style.display = "none";
  }

  function clearCanvas() {
    ctx_ui.fillStyle = "#D8ACFF";
    ctx_gray.fillStyle = "white";
    ctx_nn.fillStyle = "white";

    ctx_ui.fillRect(0, 0, canvas_ui.width, canvas_ui.height);
    ctx_gray.fillRect(0, 0, canvas_ui.width, canvas_ui.height);
    ctx_nn.fillRect(0, 0, canvas_nn.width, canvas_nn.height);
  }
};

function getResult(net, input) {
  const output = net.feedforward(input);

  return output.reduce(
    (iMax, x, i, arr) => (x[0] > arr[iMax][0] ? i : iMax),
    0
  );
}
