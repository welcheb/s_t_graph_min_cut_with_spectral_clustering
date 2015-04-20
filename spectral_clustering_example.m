%% spectral clustering example

%% clean slate
close all; clear all; clc;

%% load PNG image
img_truth = double(imread('BW.png'));

%% create normalized noisy image
sigma = 0.02;
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
imwrite(frame2im(getframe(gcf)), './figures2/Figure01.png');

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
x_shift_nbs = [-R_max:1:+R_max];
y_shift_nbs = x_shift_nbs;
nx_nbs = numel(x_shift_nbs);
ny_nbs = nx_nbs;

img_blank = zeros(nx_img, ny_img);
W = zeros(N, N);
for x_img = 1:nx_img,
    for y_img= 1:ny_img,
        
        img_blank(:) = 0.0;
        
        for x_nbs = 1:nx_nbs,
            for y_nbs = 1:ny_nbs,
                
                x_this = x_img + x_shift_nbs(x_nbs);
                y_this = y_img + y_shift_nbs(y_nbs);
                
                if (x_this>=1 && x_this<=x_img && y_this>=1 && y_this<=y_img),
                    this_I_diff_squared = ( img(x_this, y_this) - img(x_img, y_img) )^2;
                    this_X_squared = (x_this-x_img)^2 + (y_this-y_img)^2;
                    img_blank(x_this, y_this) = exp(-this_I_diff_squared/varI) * exp(-this_X_squared/varX);
                end

            end
        end
        
        idx = sub2ind([nx_img ny_img], x_img, y_img);
        W(:,idx) = img_blank(:);
        W(idx,:) = img_blank(:)';
        
    end
end

%% slow
% for idx_i = 1:N,
%     [idx_i_x, idx_i_y] = ind2sub([nx_img ny_img], idx_i);
%     for idx_j= idx_i:N, % upper triangular matrix
%         [idx_j_x, idx_j_y] = ind2sub([nx_img ny_img], idx_j);
%         
%         this_R_squared = (idx_i_x-idx_j_x)^2 + (idx_i_y-idx_j_y)^2;
%         if this_R_squared < R_max_squared,
%             this_I_diff_squared = (img(idx_i_x,idx_i_y) - img(idx_j_x,idx_j_y))^2;
%             W(idx_i,idx_j) = exp(-this_I_diff_squared/varI) * exp(-this_R_squared/varR);
%         end
%         
%         % lower triangular matrix
%         W(idx_j,idx_i) = W(idx_i,idx_j);
%         
%     end
% end

%% compute diagonal degree matrix D
D = zeros(N, N);
for idx=1:N,
    D(idx,idx) = sum(W(idx,:));
end

%% Display W
figure(2);
imagesc(W,[0.00 0.05]);
colormap(hot); axis image; title('Similarity Matrix');
impixelinfo

%% Display D
figure(3);
imagesc(D);
colormap(hot); axis image; title('Diagonal Degree Matrix');

%% Compute unnormalized Laplacian matrix
%L = D - W;

%% Compute normalized Laplacian matrix
% works better pulling out foreground with 2 clusters
D_inv_sqrt = diag(diag(D).^(-1/2));
L = D_inv_sqrt * (D-W) * D_inv_sqrt;

%% Display L
figure(4); imagesc(L); colormap(hot); axis image; title('L');

%% Compute the eigenvectors of  L
[U, ev] = eig(L, 'vector'); 

%% sort eigenvalues
[ev_sorted, ev_sorted_idx] = sort(ev);

%% plot sorted eigenvalues
figure(5); plot(ev_sorted,'b*'); grid on; title('sorted eigenvalues');
imwrite(frame2im(getframe(gcf)), './figures2/Figure05.png');

%% Keep first nCluster eigenvectors (smallest nCluster eigenvalues)
for nCluster = 2:3,
    WC = U(:,ev_sorted_idx([1:nCluster]));

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
        imwrite(frame2im(getframe(gcf)), sprintf('./figures2/Figure%02d.png', fignum) );
    elseif nCluster==3,
        fignum = 10+nCluster;
        figure(fignum);
        plot3( WC(cluster{1},1), WC(cluster{1},2), WC(cluster{1},3), 'r.',...
            WC(cluster{2},1), WC(cluster{2},2), WC(cluster{2},3), 'b.', ...
            WC(cluster{3},1), WC(cluster{3},2), WC(cluster{3},3), 'g.');
        legend('cluster 1', 'cluster 2', 'cluster 3');
        grid on;
        imwrite(frame2im(getframe(gcf)), sprintf('./figures2/Figure%02d.png', fignum) );        
    end
    
    %% display cluster images
    fignum = 20+nCluster;
    figure(fignum);
    subplot(2,3,1); imagesc(img_truth); axis image; colormap(gray); title('truth');
    subplot(2,3,2); imagesc(img); axis image; colormap(gray); title('noisy');
    
    for k_idx=1:nCluster,    
        subplot(2,3,3+k_idx); imagesc( reshape(cluster{k_idx},[nx_img ny_img]) ); 
        axis image; colormap(gray); title( sprintf('spectral clustering %d of %d', k_idx, nCluster) );
    end
    
    imwrite(frame2im(getframe(gcf)), sprintf('./figures2/Figure%02d.png', fignum) );
    
end