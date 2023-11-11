import yolov5
import os
import json
import cv2

def model_pred():
    model = yolov5.load('/home/hsumant/Desktop/programs/aNgle_rOtate/best.pt')
    dir = '/home/hsumant/Desktop/programs/aNgle_rOtate/mhka/'


    # set model parameters
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image


    # set image
    for i in os.listdir(dir):

        if (i.endswith(".jpeg")):
            # img = '/home/hsumant/Desktop/programs/aNgle_rOtate/hari/*.png'
            img = dir + i

            print(img)
            # perform inference
            results = model(img)
            # parse results
            predictions = results.pred[0]
            boxes = predictions[:, :4] # x1, y1, x2, y2
            scores = predictions[:, 4]
            categories = predictions[:, 5]

            out_json = results.pandas().xyxy[0].to_json(orient="records") 
            # print(out_json)

            out_json = json.loads(out_json)
            filtered_json = [json_obj for json_obj in out_json if json_obj.get("class") in [0, 7, 8]]
            sv_n = img.split('/')[-1].split(".")[0]
            with open(f'/home/hsumant/Desktop/programs/aNgle_rOtate/mhka/{sv_n}.json', 'w', encoding='utf-8') as f:
                json.dump(filtered_json, f, ensure_ascii=False, indent=4)
        
            # show detection bounding boxes on image
            # results.show()
            # save results into "results/" folder
        # results.save(save_dir=f'/home/hsumant/Desktop/programs/aNgle_rOtate/results/{i}.png')


if __name__ == '__main__':
    model_pred()