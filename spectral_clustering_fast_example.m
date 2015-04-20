%% spectral clustering example

%% clean slate
close all; clear all; clc;

%% load PNG image
img_truth = double(imread('CLUSTER.png'));

%% create normalized noisy image
sigma = 0.02;
sigma = 0.10;
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
imwrite(frame2im(getframe(gcf)), './figures3/Figure01.png');

%% Settings for Gaussian similarity calculations
R_max = 5; % max pixel distance

sigmaI = sqrt(0.2);
sigmaI = 0.1;
varI = sigmaI*sigmaI;
%varI = var(img(:));

sigmaX = sqrt(0.35);
sigmaX = 4.0;
varX = sigmaX * sigmaX;

%% create weighted adjacency matrix W
% square symmetric matrices with N (total pixel count) columns
tic;
x_shift_nbs = [-R_max:1:+R_max];
y_shift_nbs = x_shift_nbs;
nx_nbs = numel(x_shift_nbs);
ny_nbs = nx_nbs;

nzmax = nx_nbs * nx_nbs * N;
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
                    
                    this_I_diff_squared = ( img(x_this, y_this) - img(x_img, y_img) )^2;
                    this_X_squared = (x_this-x_img)^2 + (y_this-y_img)^2;
                    
                    W_s(W_cnt)= exp(-this_I_diff_squared/varI) * exp(-this_X_squared/varX);
                    
                    W_cnt = W_cnt + 1;
                    W_i(W_cnt) = W_j(W_cnt-1);
                    W_j(W_cnt) = W_i(W_cnt-1);
                    W_s(W_cnt) = W_s(W_cnt-1);
                end

            end
        end
                
    end
end
toc;
%% Create sparse W
tic;
W = sparse(W_i(1:W_cnt), W_j(1:W_cnt), W_s(1:W_cnt), N, N);
toc;

%% compute diagonal degree matrix D
tic;
D_ij = [1:N];
D_s = sum(W,1);
D = sparse(D_ij, D_ij, D_s, N, N);
D_inv_sqrt = sparse(D_ij, D_ij, 1./sqrt(D_s), N, N);
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
%L = D-W; % not normalized
L = D_inv_sqrt * (D-W) * D_inv_sqrt; % normalized
toc;

%% Display L
% figure(4); imagesc(L); colormap(hot); axis image; title('L');
% impixelinfo

%% Compute the first nCluster_max eigen vectors (smallest eigen values) of  L
tic;
nCluster_max = 8;
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
imwrite(frame2im(getframe(gcf)), './figures3/Figure05.png');

%% Keep first nCluster eigenvectors (smallest nCluster eigenvalues)
for nCluster = 2:nCluster_max,
    
    WC = V(:, ev_sorted_idx(1:nCluster) );

    %% perform kmeans clustering
    kidx = kmeans(WC,nCluster);
    for k_idx=1:nCluster,
        cluster{k_idx} = kidx==k_idx;  
    end
    
    %% display cluster plots
    if nCluster==2,
        fignum = 10+nCluster;
        figure(fignum);
        plot( WC(cluster{1},1), WC(cluster{1},2), 'r.', ...
            WC(cluster{2},1), WC(cluster{2},2), 'b.');
        legend('cluster 1', 'cluster 2');
        grid on;
        imwrite(frame2im(getframe(gcf)), sprintf('./figures3/Figure%02d.png', fignum) );
    elseif nCluster==3,
        fignum = 10+nCluster;
        figure(fignum);
        plot3( WC(cluster{1},1), WC(cluster{1},2), WC(cluster{1},3), 'r.',...
            WC(cluster{2},1), WC(cluster{2},2), WC(cluster{2},3), 'b.', ...
            WC(cluster{3},1), WC(cluster{3},2), WC(cluster{3},3), 'g.');
        legend('cluster 1', 'cluster 2', 'cluster 3');
        grid on;
        imwrite(frame2im(getframe(gcf)), sprintf('./figures3/Figure%02d.png', fignum) );        
    end
    
    %% display cluster images
    fignum = 20+nCluster;
    figure(fignum);
    subplot(3,3,1); imagesc(img); axis image; colormap(gray); title('noisy');
    
    for k_idx=1:nCluster,    
        subplot(3, 3, 1+k_idx); imagesc( reshape(cluster{k_idx},[nx_img ny_img]) ); 
        axis image; colormap(gray); title( sprintf('spectral clustering %d of %d', k_idx, nCluster) );
    end
    
    imwrite(frame2im(getframe(gcf)), sprintf('./figures3/Figure%02d.png', fignum) );
    
end