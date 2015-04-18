%% s-t graph min cut approximation with spectral clustering example

%% clean slate
close all; clear all; clc;

%% load PNG image
img_truth = double(imread('B.png'));

%% create normalized noisy image
sigma = 0.2;
rng(0,'v5uniform'); % seed random number generator
img = img_truth + sigma * max(img_truth(:)) * randn(size(img_truth));
img = img - min(img(:));
img = img / max(img(:));
[nx_img ny_img] = size(img);
N = nx_img * ny_img;

%% diplay original images
figure(1);
subplot(1,2,1); imagesc(img_truth); axis image; colormap(gray); title('truth');
subplot(1,2,2); imagesc(img); axis image; colormap(gray); title('input noisy image');
imwrite(frame2im(getframe(gcf)), './figures/Figure01.png');

%% 3 x 3 neighborhood kernel for use building weighted adjacency matrix (affinity/similarity)
% represents the connectivity between neighboring pixels in a typical s-t
% graph min cut procedure to segment foreground from background

nbs = [0.00 0.05 0.00; 0.05 1.00 0.05; 0.00 0.05 0.00];
[nx_nbs ny_nbs] = size(nbs);
x_shift_nbs = [-(nx_nbs-1)/2:+(nx_nbs-1)/2];
y_shift_nbs = [-(ny_nbs-1)/2:+(ny_nbs-1)/2];

%% create weighted adjacency matrix W and diagonal degree matrix D
% square symmetric matrices with N^2 + 2 columns 
% extra columns are for source s (background) and sink t (foreground) nodes
nx_W = nx_img * ny_img + 2;
W = zeros(nx_W, nx_W);
D = zeros(nx_W, nx_W);

img_blank = zeros(nx_img, ny_img);
for x_img = 1:nx_img,
    for y_img= 1:ny_img,
        
        img_blank(:) = 0.0;
        
        for x_nbs = 1:nx_nbs,
            for y_nbs = 1:ny_nbs,
                
                x_this = x_img + x_shift_nbs(x_nbs);
                y_this = y_img + y_shift_nbs(y_nbs);
                
                if (x_this>=1 && x_this<=x_img && y_this>=1 && y_this<=y_img),
                    img_blank(x_this, y_this) = nbs(x_nbs, y_nbs);
                end

            end
        end
        
        idx = sub2ind([nx_img ny_img], x_img, y_img);
        W(1:N,idx) = img_blank(:);
        W(idx,1:N) = img_blank(:)';
        
    end
end

%% create s and t col (row) to add to weighted adjacency matrix
s_col = [ 1.0-img(:); 1 ; 0];
t_col = [ img(:); 0 ; 1];
W(N+1,:) = s_col';
W(:,N+1) = s_col;
W(N+2,:) = t_col';
W(:,N+2) = t_col;

%% compute diagonal degree matrix D
D = zeros(nx_W, nx_W);
for idx=1:nx_W,
    D(idx,idx) = sum(W(idx,:));
end

%% Display W
figure(2);
imagesc(W,[0.0 1.0]);
colormap(hot); axis image; title('Similarity Matrix');

%% Display D
figure(3);
imagesc(D,[0 1]);
colormap(gray); axis image; title('Diagonal Degree Matrix');

%% Compute unnormalized Laplacian matrix
L = D - W;

%% Display L
figure(4); imagesc(L); colormap(gray); axis image; title('L');

%% Compute the eigenvectors of  L
[U, ev] = eig(L,'vector'); 

%% sort eigenvalues
[ev_sorted, ev_sorted_idx] = sort(ev);
[ev_sorted, ev_sorted_idx] = sort(abs(ev));

%% plot sorted eigenvalues
figure(5); plot(ev_sorted,'b*'); grid on; title('sorted eigenvalues');
imwrite(frame2im(getframe(gcf)), './figures/Figure05.png');

%% Keep first two eigenvectors (smallest two eigenvalues)
W = U(:,ev_sorted_idx([1:2]));

%% Plot coordinates in the 2D subspace
figure(7);
plot(W(1:N,1),W(1:N,2),'k.', W(N+1,1),W(N+1,2),'b*', W(N+2,1),W(N+2,2),'r*');
legend('pixels','s','t');
axis square;
imwrite(frame2im(getframe(gcf)), './figures/Figure07.png');

%% perform kmeans clustering
kidx = kmeans(W(1:N,:),2);
kidx_1 = kidx==1;
kidx_2 = kidx==2;

%% plot clusters
figure(8);
plot(W(kidx_1,1),W(kidx_1,2),'b.', W(kidx_2,1),W(kidx_2,2),'r.');
legend('cluster 1','cluster 2');
imwrite(frame2im(getframe(gcf)), './figures/Figure08.png');

%% plot results
img_result = reshape(kidx_1,[64 64]);
if img_result(1,1)==0,
    img_result = reshape(kidx_2,[64 64]);
end

figure(9);
subplot(2,2,1); imagesc(img_truth); axis image; colormap(gray); title('truth');
subplot(2,2,2); imagesc(img); axis image; colormap(gray); title('noisy');
subplot(2,2,3); imagesc(img_result); axis image; colormap(gray); title('result - spectral clustering');
subplot(2,2,4); imagesc(img>graythresh(img)); axis image; colormap(gray); title('result - Otsu');

imwrite(frame2im(getframe(gcf)), './figures/Figure09.png');