import argparse

def main():
    parser = argparse.ArgumentParser(description='arguments passed to VideoProcessor')
    parser.add_argument("-i", "--input", type=str, help="data root for source video and images")
    parser.add_argument("-o", "--output", type=str, help="folder to store the generated video and images")
    parser.add_argument("--file_path", type=str, help="file path which records compositon fg and bg")
    parser.add_argument("--phase", type=str, help="train or test")
    parser.add_argument("--image_only", action="store_true", help="whether to generate image datapoints only")
    parser.add_argument("--video_only", action="store_true", help="whether to generate video datapoints only")
    parser.add_argument("--bg_blur", action="store_true", help="whether to blue the background video")
    parser.add_argument("--n_proc", type=int, default=10, help="number of process for multiprocessing")
    parser.add_argument("--classes", nargs="+", type=str, help="classes of foreground images/videos to be used for generation (easy, medium, hard)")
    parser.add_argument("--max_size_from_image", type=int, default = 4000, help="max number of datapoints generated using the foreground images")
    parser.add_argument("--max_size_from_video", type=int, default = 1000, help="max number of datapoints generated using the foreground videos")

    args = parser.parse_args()

    from video_processor import VideoProcessor, VIDEO, IMAGE
    vp = VideoProcessor(data_root=args.input, save_root=args.output, file_list_path=args.file_path, phase=args.phase, \
                            classes_used = args.classes, save_mode=IMAGE, bg_blur=args.bg_blur)

    if args.image_only:
        vp.batch_image(n_proc = args.n_proc, max_n = args.max_size_from_image)
    elif args.video_only:
        vp.batch_video(n_proc = args.n_proc, max_n = args.max_size_from_video)
    else:
        vp.batch_image(n_proc = args.n_proc, max_n = args.max_size_from_image)
        vp.batch_video(n_proc = args.n_proc, max_n = args.max_size_from_video)

if __name__ == "__main__":
    main()
