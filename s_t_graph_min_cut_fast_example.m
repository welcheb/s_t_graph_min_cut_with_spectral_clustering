%% FAST s-t graph min cut approximation with spectral clustering example

%% clean slate
close all; clear all; clc;

%% load PNG image
img_truth = double(imread('CLUSTER.png'));

%% create normalized noisy image
sigma = 0.40;
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
imwrite(frame2im(getframe(gcf)), './figures4/Figure01.png');

%% create weighted adjacency matrix W
% square symmetric matrices with N (total pixel count) columns
tic;

nbs = [0.00 0.05 0.00; 0.05 1.00 0.05; 0.00 0.05 0.00];
[nx_nbs ny_nbs] = size(nbs);
x_shift_nbs = [-(nx_nbs-1)/2:+(nx_nbs-1)/2];
y_shift_nbs = [-(ny_nbs-1)/2:+(ny_nbs-1)/2];

nzmax = nx_nbs * nx_nbs * N + 4*N;
W_i = zeros(nzmax,1);
W_j = zeros(nzmax,1);
W_s = zeros(nzmax,1);
W_cnt = 0;
for x_img = 1:nx_img,
    for y_img= 1:ny_img,
                
        for x_nbs = 1:nx_nbs,
            for y_nbs = 1:ny_nbs,
                
                x_this = x_img + x_shift_nbs(x_nbs);
                y_this = y_img + y_shift_nbs(y_nbs);
                
                if (x_this>=1 && x_this<=x_img && y_this>=1 && y_this<=y_img),
                    
                    W_cnt = W_cnt + 1;
                    
                    W_i(W_cnt) = (y_img - 1) * nx_img + x_img;
                    
                    W_j(W_cnt) = (y_this - 1) * nx_img + x_this;
                                        
                    W_s(W_cnt)= nbs(x_nbs, y_nbs);
                    
                    W_cnt = W_cnt + 1;
                    W_i(W_cnt) = W_j(W_cnt-1);
                    W_j(W_cnt) = W_i(W_cnt-1);
                    W_s(W_cnt) = W_s(W_cnt-1);
                end

            end
        end
                
    end
end

% s and t entries
s_links = [ 1.0-img(:); 1 ; 0];
t_links = [ img(:); 0 ; 1];
for idx = 1:(N+2),
    
    % s
    W_cnt = W_cnt + 1;
    W_i(W_cnt) = N+1; 
    W_j(W_cnt) = idx;
    W_s(W_cnt) = s_links(idx);
    
    W_cnt = W_cnt + 1;
    W_i(W_cnt) = W_j(W_cnt-1);
    W_j(W_cnt) = W_i(W_cnt-1);
    W_s(W_cnt) = W_s(W_cnt-1);

    % t
    W_cnt = W_cnt + 1;
    W_i(W_cnt) = N+2; 
    W_j(W_cnt) = idx;
    W_s(W_cnt) = t_links(idx);
    
    W_cnt = W_cnt + 1;
    W_i(W_cnt) = W_j(W_cnt-1);
    W_j(W_cnt) = W_i(W_cnt-1);
    W_s(W_cnt) = W_s(W_cnt-1);    
end
toc;
%% Create sparse W
tic;
W = sparse(W_i(1:W_cnt), W_j(1:W_cnt), W_s(1:W_cnt), N+2, N+2);
toc;

%% compute diagonal degree matrix D
tic;
D_ij = [1:N+2];
D_s = sum(W,1);
D = sparse(D_ij, D_ij, D_s, N+2, N+2);
D_inv_sqrt = sparse(D_ij, D_ij, 1./sqrt(D_s), N+2, N+2);
toc;

%% Display W
% figure(2);
% imagesc(W,[0.00 0.05]);
% colormap(hot); axis image; title('Similarity Matrix');
% impixelinfo

%% Display D
% figure(3);
% imagesc(D);
% colormap(hot); axis image; title('Diagonal Degree Matrix');
% impixelinfo

%% Compute Laplacian matrix
tic;
L = D-W; % not normalized
%L = D_inv_sqrt * (D-W) * D_inv_sqrt; % normalized
toc;

%% Display L
% figure(4); imagesc(L); colormap(hot); axis image; title('L');
% impixelinfo

%% Compute the first nCluster_max eigen vectors (smallest eigen values) of  L
tic;
nCluster_max = 2;
if issymmetric(L),
    [V, E] = eigs(L, nCluster_max, 'SA');
else
    [V, E] = eigs(L, nCluster_max, 'SM');
    %[V, E] = eigs(L, nCluster_max, 'SR');
end

%% Handle complex V, E
if ~isreal(V),
    V = real(V);
    E = real(E);
end

%% Sort eigen values
ev = diag(E);
[ev_sorted, ev_sorted_idx] = sort(ev);
toc;

%% plot sorted eigenvalues
figure(5); plot(1:nCluster_max, ev_sorted,'b*'); grid on;
title( sprintf('first %d eigenvalues', nCluster_max) );
imwrite(frame2im(getframe(gcf)), './figures4/Figure05.png');

%% Keep first two eigenvectors (smallest two eigenvalues)
W = V(:,ev_sorted_idx([1:2]));

%% Plot coordinates in the 2D subspace
figure(7);
plot(W(1:N,1),W(1:N,2),'k.', W(N+1,1),W(N+1,2),'b*', W(N+2,1),W(N+2,2),'r*');
legend('pixels','s','t');
axis square;
imwrite(frame2im(getframe(gcf)), './figures4/Figure07.png');

%% perform kmeans clustering
kidx = kmeans(W(1:N,:),2);
kidx_1 = kidx==1;
kidx_2 = kidx==2;

%% plot clusters
figure(8);
plot(W(kidx_1,1),W(kidx_1,2),'b.', W(kidx_2,1),W(kidx_2,2),'r.');
legend('cluster 1','cluster 2');
imwrite(frame2im(getframe(gcf)), './figures4/Figure08.png');

%% plot results
img_result = reshape(kidx_1,[nx_img ny_img]);
if img_result(1,1)==0,
    img_result = reshape(kidx_2,[nx_img ny_img]);
end

figure(9);
subplot(2,2,1); imagesc(img_truth); axis image; colormap(gray); title('truth');
subplot(2,2,2); imagesc(img); axis image; colormap(gray); title('noisy');
subplot(2,2,3); imagesc(img_result); axis image; colormap(gray); title('result - spectral clustering');
subplot(2,2,4); imagesc(img>graythresh(img)); axis image; colormap(gray); title('result - Otsu');

imwrite(frame2im(getframe(gcf)), './figures4/Figure09.png');