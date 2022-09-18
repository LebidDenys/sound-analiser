import os
import time


def download_video(label, out_folder, inp_folder):
    print('Downloading for class:{}'.format(label))

    if not os.path.exists(os.path.join(out_folder, label)):
        os.mkdir(os.path.join(out_folder, label))

    if not os.path.exists(os.path.join(out_folder, 'problem_vids')):
        os.mkdir(os.path.join(out_folder, 'problem_vids'))

    f = open(os.path.join(out_folder, 'problem_vids', label + '.txt'), "w")
    f.close()
    out_file = os.path.join(out_folder, label, 'temp' + '.%' + '(ext)s')
    with open(os.path.join(inp_folder, label, 'full_labels.txt')) as txtfile:
        num_lines = sum(1 for line in open(os.path.join(inp_folder, label, 'full_labels.txt')))
        vids = 1
        for lines in txtfile.readlines():
            linesPart = lines.split(', ')
            url = ' https://www.youtube.com/watch?v=' + linesPart[1]

            print('-------------Downloading:{}/{}----------------'.format(vids, num_lines))
            print(url)
            start_time = time.time()

            youtube_dl_options = '\'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best\''

            cmd = 'yt-dlp --quiet --no-warnings -f ' + \
                  + '--extract-audio --audio-format wav' + \
                  youtube_dl_options + ' -o ' + '\'' + out_file + '\'' + ' ' + url
            # print(cmd)
            download = os.system(cmd)
            if download != 0:
                f = open(os.path.join(out_folder, 'problem_vids', label + '.txt'), "a")
                f.write(", ".join(linesPart))
                f.close()
            else:
                os.system(
                    'ffmpeg -nostats -loglevel 0 -i ' + os.path.join(out_folder, label) + '/temp.mp4 -ss ' + linesPart[
                        2] + ' -strict -2 -t ' + str(
                        float(linesPart[3]) - float(linesPart[2])) + ' ' + os.path.join(out_folder, label) + '/' + str(
                        "%03d" % (vids,)) + '.mp4')
                os.remove(os.path.join(out_folder, label, 'temp.mp4'))
            vids += 1
            print('Time Taken in Seconds:{}'.format(time.time() - start_time))
