data=load("out8_11c.txt");
rows=54;
colunms=71;
X=reshape(data(:,1),rows,colunms);
Y=reshape(data(:,2),rows,colunms);
Zc=reshape(data(:,3),rows,colunms);
figure
contour(X,Y,Zc,30);

data=load("out8_11g.txt");
Zg=reshape(data(:,3),rows,colunms);
figure
contour(X,Y,Zg,30);

disp(max(max(abs(Zc-Zg))))

