from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from io import BytesIO
import numpy as np
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)

app = FastAPI()

# load models
refine_net = load_refinenet_model(cuda=True)
craft_net = load_craftnet_model(cuda=True)



@app.post("/detect_text/")
async def detect_text(image_file: UploadFile = File(...)):
    # read uploaded image
    contents = await image_file.read()
    image = Image.open(BytesIO(contents))

    # perform prediction
    prediction_result = get_prediction(
        image=image,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.7,
        link_threshold=0.6,
        low_text=0.35,
        cuda=True,
        long_size=1280
    )

    # export detected text regions
    regions = prediction_result["boxes"]
    exported_file_paths = export_detected_regions(
        image=image,
        regions=regions,
        output_dir='./static',
        rectify=True
    )

    # export heatmap, detection points, box visualization
    export_extra_results(
        image=image,
        regions=regions,
        heatmaps=prediction_result["heatmaps"],
        output_dir='./static'
    )

    # generate HTML response to display results on the web
    response_html = "<html><body>"
    response_html += f"<img src='/static/{image_file.filename}'><br>"
    for i, exported_file_path in enumerate(exported_file_paths):
        response_html += f"<img src='/static/{exported_file_path.stem}_text_region.png'><br>"
    response_html += "</body></html>"

    return HTMLResponse(content=response_html)

# unload models from gpu
empty_cuda_cache()



def get_prediction(image, craft_net, refine_net, text_threshold, link_threshold, low_text, cuda, long_size):
    img = None
    if isinstance(image, str):
        img = read_image(image)
    elif isinstance(image, np.ndarray):
        img = image.copy()
    elif isinstance(image, Image.Image):
        img = np.array(image.convert('RGB'))

    if img is not None:
        # resize image
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(img, long_size, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
        ratio_h = ratio_w = 1 / target_ratio

        # prepare data
        x = img_resized.astype(np.float32) / 255.
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

        # forward pass
        with torch.no_grad():
            if cuda:
                x = x.cuda()
            y, feature = craft_net(x)
            score_text = y[0,:,:,0].cpu().data.numpy()
            score_link = y[0,:,:,1].cpu().data.numpy()

            # refine link
            refine_net.eval()
            y_refiner = refine_net(y, feature)
            score_link = y_refiner[0,:,:,0].cpu().data.numpy()

        # Post-processing
        boxes, polys = get_det_boxes(score_text, score_link, text_threshold, link_threshold, low_text, False, poly=False)
        boxes = boxes.reshape((-1, 4))

        # unresize boxes
        boxes[:,[0,2]] /= ratio_w
        boxes[:,[1,3]] /= ratio_h

        return {"boxes": boxes, "polys": polys, "heatmaps": [score_text, score_link], "size_heatmap": size_heatmap}