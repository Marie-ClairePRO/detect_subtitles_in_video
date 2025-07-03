import cv2 as cv
import easyocr
import os
import ffmpeg
import Levenshtein
import numpy as np
import argparse

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def similarity(text1, text2):
    #1.0 = same text
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
    distance = Levenshtein.distance(text1, text2)
    max_len = max(len(text1), len(text2)) 
    return 1 - (distance / max_len)

def res_are_similar(res1, res2, similarity_thresh=0.7):
    similarities_list = [(similarity(t1,t2)>similarity_thresh and is_same_bbox(bb1,bb2)) 
                         for (bb1,t1,_) in res1 for (bb2,t2,_) in res2]
    return (any(similarities_list) 
            or similarity(create_multiline_text_from_res(res1), create_multiline_text_from_res(res2))>similarity_thresh)

def create_multiline_text_from_res(ocr_results, min_confidence=0.3):
    filtered = [ (bbox, text, prob) for (bbox, text, prob) in ocr_results if prob >= min_confidence]
    
    if not filtered:
        return ""
    
    Y_HEIGHTS = [abs(res[0][2][1] - res[0][0][1]) for res in filtered]
    Y_HEIGHT = int(sum(Y_HEIGHTS)/len(Y_HEIGHTS))
    entries = []
    for bbox, text, prob in filtered:
        #bbox contains 4 coordinates of sides or rectangle -> we want x_left and y_middle
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        x_left = min(x_coords)
        y_center = sum(y_coords) / len(y_coords)
        entries.append({
            'bbox': bbox,
            'text': text,
            'prob': prob,
            'x_left': x_left,
            'y_center': y_center
        })

    #sort given height
    entries.sort(key=lambda x: x['y_center'])

    lines = []
    current_line = []
    current_y = entries[0]['y_center']

    for entry in entries:
        #considered same line because same ordonnate
        if abs(entry['y_center'] - current_y) <= Y_HEIGHT//2:
            current_line.append(entry)
        else:
            #end of line, sort by abscisse
            current_line.sort(key=lambda x: x['x_left'])
            lines.append(current_line)
            #new line to initialize
            current_line = [entry]
            current_y = entry['y_center']

    if current_line:
        current_line.sort(key=lambda x: x['x_left'])
        lines.append(current_line)

    final_lines = []

    for line in lines:
        line_text = " ".join([entry['text'] for entry in line])
        final_lines.append(line_text)

    assembled_text = "\n".join(final_lines)

    return assembled_text

def close_points(p1, p2, CLOSE):
    return abs(p1[0] - p2[0]) < CLOSE and abs(p1[1] - p2[1]) < CLOSE 

def same_line_points(b1_top, b2_top, b1_bottom, b2_bottom, CLOSE):
    return abs(b1_top[1] - b2_top[1]) < CLOSE and abs(b1_bottom[1] - b2_bottom[1]) < CLOSE

def b1_in_b2(bbox1, bbox2, CLOSE):
    return ((bbox1[0][0] > bbox2[0][0] + CLOSE and bbox1[2][0] < bbox2[2][0] + CLOSE) or
            (bbox1[0][0] > bbox2[0][0] - CLOSE and bbox1[2][0] < bbox2[2][0] - CLOSE))

def b1_above_b2(bbox1, bbox2, CLOSE):
    return (bbox1[0][0] > bbox2[0][0] + CLOSE and bbox1[0][0] < bbox2[2][0] - CLOSE and bbox1[2][0] > bbox2[2][0] + CLOSE)

def is_strictly_overlapping(bbox1, bbox2, CLOSE):
    return ((b1_above_b2(bbox1, bbox2, CLOSE) or b1_above_b2(bbox2, bbox1, CLOSE)) and
            same_line_points(bbox1[0], bbox2[0], bbox1[2], bbox2[2], CLOSE))

def is_same_bbox(bbox1, bbox2, CLOSE=10):
    return close_points(bbox1[0], bbox2[0], CLOSE) and close_points(bbox1[2], bbox2[2], CLOSE)

def is_included(bbox1, bbox2, CLOSE):
    return same_line_points(bbox1[0], bbox2[0], bbox1[2], bbox2[2], CLOSE) and b1_in_b2(bbox1, bbox2, CLOSE)

def mix_results(results):
    combined = [res for result in results for res in result]
    Y_HEIGHTS = [abs(res[0][2][1] - res[0][0][1]) for res in combined]
    CLOSE = int(sum(Y_HEIGHTS)/len(Y_HEIGHTS) / 2)
    final_results = []

    while combined:
        add = True
        ref = combined.pop(0)
        ref_bbox, ref_text, ref_prob = ref
        to_remove = []
        
        for i, (bbox, text, prob) in enumerate(combined):
            if is_same_bbox(ref_bbox, bbox, CLOSE):
                # if overlapping keep best prob
                if prob > ref_prob:
                    ref = (bbox, text, prob)
                to_remove.append(i)
            elif is_included(ref_bbox, bbox, CLOSE):
                # if included, keep bigger
                #ref = (bbox, text, prob)
                add = False
                break
                #to_remove.append(i)
            elif is_included(bbox, ref_bbox, CLOSE):
                to_remove.append(i)
            elif is_strictly_overlapping(bbox, ref_bbox, CLOSE):
                print(text, "-- is overlapping with --", ref_text)
                add = False
                #TODO
        
        for index in sorted(to_remove, reverse=True):
            combined.pop(index)
    
        if add:
            final_results.append(ref)
    
    return final_results

def improve_subs(subtitles, similarity_threshold=0.8):
    if not subtitles:
        return []

    merged = []
    prev = subtitles[0]

    for current in subtitles[1:]:
        if res_are_similar(prev['res'], current['res'], similarity_thresh=similarity_threshold):                
            prev['end'] = current['end']
            prev["res"] = mix_results([prev["res"], current["res"]])
        elif current["end"]-current["start"]>0.1: 
            merged.append(prev)
            prev = current

    merged.append(prev)
    return merged

def frame_duration_of(video_path):
    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    fps = eval(video_stream['r_frame_rate'])
    frame_duration = 1 / fps
    return frame_duration

def confirm_and_delete(video_path, frames_folder):

    confirmation = input(f"Folder '{frames_folder}' already exists. Delete its content and generate new frames ? (y/n) ")
    if confirmation.lower() == 'y':         
        import shutil
        shutil.rmtree(frames_folder)
        os.makedirs(frames_folder)
        print(f"Former frames in folder '{frames_folder}' are all deleted.")
        print(f"Extracting frames from {video_path} using ffmpeg...")
        (
        ffmpeg
        .input(video_path)
        .output(f"{frames_folder}/frame_%06d.jpg", qscale=2)
        .global_args('-hide_banner', '-loglevel', 'error')
        .run(overwrite_output=True)
        )
    else:
        print(f"Extracting subtitles from already existing frames in '{frames_folder}'. \n")

def extract_subs_of_all_frames(frames_folder, frame_duration, reader, args, similarity_thresh):
    #initialization
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg') or f.endswith('.png')])
    subs = []
    current_res = None

    # LOOP ON FRAMES TO CREATE SUBS
    for i, frame_file in enumerate(frame_files):
        frame = cv.imread(os.path.join(frames_folder, frame_file))
        timestamp = i * frame_duration
        h, w = frame.shape[:2]
        frame = frame[int(args.cut_top *h) : int((1-args.cut_bottom)* h),
                      int(args.cut_left*w) : int((1-args.cut_right) * w)]
        result = reader.readtext(frame)

        #result is array of (bbox, text, prob)
        #bbox is 4-size array containing 2-size arrays [[x1,y1],...] in order topleft-topright-bottomright-bottomleft
        #first : filter by prob and size
        result = [r for r in result if r[2]>0.3]

        #if last frame does not have subtitles, initialize new subs or stay empty
        if current_res is None:
            if result != []:
                current_res = {'start': timestamp, 'end': timestamp, 'res': result}

        #if last frame has subtitles
        else:
            #and current ones are the continuation of previous ones -> 
            if res_are_similar(result,current_res["res"], similarity_thresh=similarity_thresh):
                current_res['end'] = timestamp
                current_res["res"] = mix_results([result, current_res["res"]])

            #if not similar, then new sub : end current one and initialize new one
            else:
                subs.append(current_res)
                print("\n", create_multiline_text_from_res(current_res["res"]), "\n")
                #new subs
                if result != []:
                    current_res = {'start': timestamp, 'end': timestamp, "res": result}
                #or empty frame
                else:
                    current_res = None

    #last one if not added (ends on last frame)
    if current_res:
        subs.append(current_res)

    return subs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, help='Path to the input video', required=True)
    parser.add_argument('--frames_folder', type=str, default="frames" , help='Path to the generated frames')
    parser.add_argument('--lang', nargs="+", default=["fr", "en", "de"], help='List of language(s) in video')
    parser.add_argument('--cut_top', type=float, default=0.5, help='Ratio of image where you find subtitles. Cut half (default) is 0.5')
    parser.add_argument('--cut_right', type=float, default=0, help='Ratio of image where you find subtitles. Cut half is 0.5')
    parser.add_argument('--cut_bottom', type=float, default=0, help='Ratio of image where you find subtitles. Cut half is 0.5')
    parser.add_argument('--cut_left', type=float, default=0, help='Ratio of image where you find subtitles. Cut half is 0.5')
    parser.add_argument('--difficulty_level', type=float, default=1 , help='Estimated difficulty from 1 to 5')
    args = parser.parse_args()
    
    video_path= args.video
    frames_folder = args.frames_folder 
    lang = args.lang
    DIFFICULTY_LVL = args.difficulty_level
    SIMILARITY_THRESH = round(0.75 - DIFFICULTY_LVL * 0.05,2)
    assert SIMILARITY_THRESH > 0.2, "difficulty level too high. Try lower or equal to 5."

    print(f"Extracting subtitles from {video_path} in {lang}, with similarity threshold {SIMILARITY_THRESH}")

    reader = easyocr.Reader(lang)

    #video_path exists but not frames folder -> create it to fill with frames
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)
    
    #video_path exists but the frames folder is not empty -> ask to delete its content
    elif len(os.listdir(frames_folder)) > 0:
        confirm_and_delete(video_path, frames_folder)

    frame_duration = frame_duration_of(video_path)
    subs = extract_subs_of_all_frames(frames_folder, frame_duration, reader, args, SIMILARITY_THRESH)

    all_probs = [box[2] for sub in subs for box in sub["res"]]
    print("OVERALL VALIDITY --------" , sum(all_probs)/len(all_probs))

    #Make subs better by merging similar ones
    subs = improve_subs(subs, similarity_threshold=SIMILARITY_THRESH*1.2)

    output_srt = video_path.split(".")[0]+".srt"

    print(f"Writing srt file : ({output_srt})...")
    with open(output_srt, 'w', encoding='utf-8') as f:
        for idx, sub in enumerate(subs, 1):
            f.write(f"{idx}\n")
            start = format_timestamp(sub['start'])
            end = format_timestamp(sub['end'] + frame_duration)  # ajoute une frame à la fin
            f.write(f"{start} --> {end}\n")
            f.write(f"{create_multiline_text_from_res(sub['res'])}\n\n")

    print("Finished.")


if __name__ == '__main__':
    main()
