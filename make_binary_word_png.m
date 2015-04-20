%% clean slate
close all; clear all; clc;

%% make binary word png
size_img = 64;
word = 'CLUSTER'; f = 0.56;
word = 'BW'; f = 0.64;
nletters = numel(word);
size_font = 54;

%% initialize image
img = ones(size_img, ceil(size_img * f * nletters) );

%% make an image the same size and put text in it 
hf = figure('color', 'white', 'units', 'normalized', 'position',[.1 .1 .8 .8]); 
image(ones(size(img))); 
set(gca, 'units', 'pixels', 'position', [5 5 size(img,2)-1 size(img,1)-1], 'visible', 'off')
text('units', 'pixels', 'position', [size_img/2 - size_font/3 size_img/2 - size_font/9], ...
    'fontsize', size_font, 'fontweight', 'bold', 'FontName', 'Courier', 'string', word); 

%% capture the text 
F = getframe(gca); 
close(hf) 

%% find indices of text
c = F.cdata(:,:,1);
[i,j] = find(c==0);
ind = sub2ind(size(img),i,j);
img(ind) = 0.0;

%% write to png
imwrite(img, sprintf('%s.png', word) );
