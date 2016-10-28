%% Read and show images
clear
clc

% Read the training data
M = csvread('train.csv');

% % Delete Outliers
% N = 42000/16; 
% L = 1:N-1;
% L = L+12;
% M(L,:) = []; 
% % 
% % % Write a new csv
% csvwrite('M.csv',M)

Digits = M(:,1);
Pixels = M(:,2:end);

% Test for digits showing
n = 1;
% L = (n-1)*16 + 13;
ind = (n-1)*16+1:n*16;
pixels = Pixels(ind,:);
digits = Digits(ind); 
% digits';

m1 = pixels(1,:); m1 = reshape(m1,28,28); m1 = m1';
m2 = pixels(2,:); m2 = reshape(m2,28,28); m2 = m2';
m3 = pixels(3,:); m3 = reshape(m3,28,28); m3 = m3';
m4 = pixels(4,:); m4 = reshape(m4,28,28); m4 = m4';

m5 = pixels(5,:); m5 = reshape(m5,28,28); m5 = m5';
m6 = pixels(6,:); m6 = reshape(m6,28,28); m6 = m6';
m7 = pixels(7,:); m7 = reshape(m7,28,28); m7 = m7';
m8 = pixels(8,:); m8 = reshape(m8,28,28); m8 = m8';

m9 = pixels(9,:); m9 = reshape(m9,28,28); m9 = m9';
m10 = pixels(10,:); m10 = reshape(m10,28,28); m10 = m10';
m11 = pixels(11,:); m11 = reshape(m11,28,28); m11 = m11';
m12 = pixels(12,:); m12 = reshape(m12,28,28); m12 = m12';

m13 = pixels(13,:); m13 = reshape(m13,28,28); m13 = m13';
m14 = pixels(14,:); m14 = reshape(m14,28,28); m14 = m14';
m15 = pixels(15,:); m15 = reshape(m15,28,28); m15 = m15';
m16 = pixels(16,:); m16 = reshape(m16,28,28); m16 = m16';

subplot(4,4,1)
image(m1)
k = ind(1);
title(Digits(k))

subplot(4,4,2)
image(m2)
k = ind(2);
title(Digits(k))

subplot(4,4,3)
image(m3)
k = ind(3);
title(Digits(k))

subplot(4,4,4)
image(m4)
k = ind(4);
title(Digits(k))


subplot(4,4,5)
image(m5)
k = ind(5);
title(Digits(k))

subplot(4,4,6)
image(m6)
k = ind(6);
title(Digits(k))

subplot(4,4,7)
image(m7)
k = ind(7);
title(Digits(k))

subplot(4,4,8)
image(m8)
k = ind(8);
title(Digits(k))

subplot(4,4,9)
image(m9)
k = ind(9);
title(Digits(k))

subplot(4,4,10)
image(m10)
k = ind(10);
title(Digits(k))

subplot(4,4,11)
image(m11)
k = ind(11);
title(Digits(k))

subplot(4,4,12)
image(m12)
k = ind(12);
title(Digits(k))


subplot(4,4,13)
image(m13)
k = ind(13);
title(Digits(k))

subplot(4,4,14)
image(m14)
k = ind(14);
title(Digits(k))

subplot(4,4,15)
image(m15)
k = ind(15);
title(Digits(k))

subplot(4,4,16)
image(m16)
k = ind(16);
title(Digits(k))

