## Data Description
As per[1] PS-Battles dataset which is gathered from a large community of image manipulation enthusiasts and provides a basis for media derivation and manipulation detection in the visual domain. The dataset consists of 102'028 images grouped into 11'142 subsets, each containing the original image as well as a varying number of manipulated derivatives.

Each original image has a correpsonding sub directory (having the same image name) under the photoshopped directory that contains the mainipulated version of this image.

![image](https://user-images.githubusercontent.com/34656794/144100309-5f2c88dd-6f63-4b32-bf0b-ce3d107a4cd5.png)

As per the above example, photos 100c1k.jpg, 100d24.jpg and 100jh1.jpg under the originals folder will have a sub directory with the same name where corresponding photoshopped images are located.

The following is an example of original photo "141vnd.jpg" (on the top) extracted from the original folder and its correspondig photoshopped versions under /photoshops/141vnd/

![SampleData](https://user-images.githubusercontent.com/34656794/144101189-ca555c99-78e8-440e-a3b5-17335e579ca4.png)

The original and photoshopped files contains mainly jpg images expect for few images in png format. 

## Downloading the data
Data can be downloaded as per the intsruction here: https://github.com/dbisUnibas/PS-Battles
The github repo will provide intsructions to download two files (one for original and one for photoshopped images), For Ununtu (which is what we used in this project) the referenced Repo provides a download.sh file to download the data


[1] https://arxiv.org/abs/1804.04866


## Checking the data
After the download, the counts of downloaded files can be retrieved via the following commands (worked with ubuntu):

```
/originals$ ls | wc -l

```
Running this should show the following (might change if more/less erros found during the download process):
11120

To check the files under the photoshops folder, you can run the following loop:
```
/photoshops$find . -maxdepth 1 -type d | while read -r dir
> do printf "%s:\t" "$dir"; find "$dir" -type f | wc -l; done

```
This will provide the total as well as image per each subdirectory.
Running this should show the following (might change if more/less erros found during the download process):
90870
./10092l:       4
./100c1k:       1
./100d24:       6
./100jh1:       15
./100qo2:       11
./100yc6:       3
./101453:       3
