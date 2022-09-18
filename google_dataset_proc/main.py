from helper_function import download_video

if __name__ == '__main__':
    input_path = 'google_dataset_proc/input/'
    output_path = 'google_dataset_proc/output/'

    download_video('boom', output_path, input_path)
