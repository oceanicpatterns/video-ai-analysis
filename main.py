import os
from video_processor import process_video
from report_generator import generate_txt_report

def main():
    video_folder = 'videos/'
    analysis_results = {}

    for video_file in os.listdir(video_folder):
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(video_folder, video_file)
            print(f"Analyzing {video_file}...")
            try:
                result = process_video(video_path)
                analysis_results[video_file] = result
            except Exception as e:
                print(f"Error analyzing {video_file}: {str(e)}")
                continue

    generate_txt_report(analysis_results)
    print("Analysis complete. Report generated.")

if __name__ == "__main__":
    main()
