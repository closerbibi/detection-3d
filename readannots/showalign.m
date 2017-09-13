function showalign(points3d, original_corners, R, xmin, ymin, zmin, xmax, ymax, zmax)
    figure
    pcshow(points3d);xlabel('X(m)')
    ylabel('Y(m)')
    zlabel('Z(m)')
    title('Original Point Cloud')
    % box
    hold on;
    for i = 1:length(original_corners)
        X1 = [original_corners{i}(1:4,1); original_corners{i}(1,1)];
        Y1 = [original_corners{i}(1:4,2); original_corners{i}(1,2)];
        Z1 = [original_corners{i}(1:4,3); original_corners{i}(1,3)];
        X2 = [original_corners{i}(5:8,1); original_corners{i}(5,1)];
        Y2 = [original_corners{i}(5:8,2); original_corners{i}(5,2)];
        Z2 = [original_corners{i}(5:8,3); original_corners{i}(5,3)];
        plot3(X1,Y1,Z1);   % draw a square in the xy plane with z = 0
        plot3(X2,Y2,Z2); % draw a square in the xy plane with z = 1
        set(gca,'View',[-28,35]); % set the azimuth and elevation of the plot
        for k=1:length(X1)-1
           plot3([X1(k);X2(k)],[Y1(k);Y2(k)],[Z1(k);Z2(k)]);
        end
    end
    figure
    pcshow(points3d*R);xlabel('X(m)')
    ylabel('Y(m)')
    zlabel('Z(m)')
    title('new Point Cloud')
    % box
    hold on;
    for i = 1:length(original_corners)
        X1 = [xmin(i),xmin(i),xmax(i),xmax(i)];
        Y1 = [ymin(i),ymax(i),ymax(i),ymin(i)];
        Z1 = [zmin(i),zmin(i),zmin(i),zmin(i)];
        X2 = [xmin(i),xmin(i),xmax(i),xmax(i)];
        Y2 = [ymin(i),ymax(i),ymax(i),ymin(i)];
        Z2 = [zmax(i),zmax(i),zmax(i),zmax(i)];
        plot3(X1,Y1,Z1);   % draw a square in the xy plane with z = 0
        plot3(X2,Y2,Z2); % draw a square in the xy plane with z = 1
        set(gca,'View',[-28,35]); % set the azimuth and elevation of the plot
        for k=1:length(X1)-1
           plot3([X1(k);X2(k)],[Y1(k);Y2(k)],[Z1(k);Z2(k)]);
        end
    end
    
