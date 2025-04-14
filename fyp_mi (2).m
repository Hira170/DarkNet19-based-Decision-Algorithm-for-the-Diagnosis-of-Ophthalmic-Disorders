
Clear all
close all
clc
numfiles=3726;
46ilename = ‘E:\FYP\Datasets\ODIR\PICTURES\final_dataset’;
fori=1:numfiles
RGB=imread(46ilena(‘z (%d).jpg’, i));
LAB = rgb2lab(RGB);
L = LAB(:,:,1)/100;
L = adapthisteq(L,’NumTiles’,[8 8],’ClipLimit’,0.005);
LAB(:,:,1) = L*100;
J = lab2rgb(LAB);
imshow(J)
46ilename = fullfile(46ilename,46ilena(‘e%d.jpg’,i));
imwrite(J,46ilename);
end


Imds =
datastore(‘E:\FYP\Datasets\ODIR\PICTURES\final_dataset’,’Includ
eSubfolders’,true,’LabelSource’,’foldernames’);
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,’randomized’);

numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
fori = 1:16
subplot(4,4,i)
I = readimage(imdsTrain,idx(i));
imshow(I)
end
net =darknet19;
inputSize = net.Layers(1).InputSize;
analyzeNetwork(net)
augimdsTrain =
augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);
layer = ‘leaky9’;
featuresTrain = activations(net,augimdsTrain,layer);
featuresTest = activations(net,augimdsTest,layer);
whosfeaturesTrain
featuresTrain = squeeze(mean(featuresTrain,[1 2]))’;
featuresTest = squeeze(mean(featuresTest,[1 2]))’;
whosfeaturesTrain
Ytrain = imdsTrain.Labels;
Ytest = imdsTest.Labels;
T=importdata(‘final.xlsx’);
C=T.data;
F0=C(1:741, ; %Minority Class

F1=C(742:end, ; %Majority Class
numEx0=741; %Minority Class
numEx1=1086; %Majority Class
labels0 = false([numEx0 1]);
labels1 = true ([numEx1 1]);
adasyn_features = [F0; F1];
adasyn_labels = [labels0 ; labels1];
adasyn_beta = []; %let ADASYN choose
default
adasyn_kDensity = []; %let ADASYN choose
default
adasyn_kSMOTE = []; %let ADASYN choose
default
adasyn_featuresAreNormalized = false; %false lets ADASYN handle normalization
[out_featuresSyn, out_labelsSyn] = ADASYN(adasyn_features,
adasyn_labels, adasyn_beta, adasyn_kDensity, adasyn_kSMOTE,
adasyn_featuresAreNormalized);
