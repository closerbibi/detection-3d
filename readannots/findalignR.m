function R = findalignR(points3d)
    %% find plane
    % Set the maximum point-to-plane distance (2cm) for plane fitting.
    maxDistance = 0.02;
    % Set the normal vector of the plane.
    referenceVector = [0,0,1];
    % Set the maximum angular distance to 5 degrees.
    maxAngularDistance = 5;

    % Detect the first plane, the table, and extract it from the point cloud.
    ptCloud = pointCloud(points3d);
    [model1,inlierIndices,outlierIndices] = pcfitplane(ptCloud,...
            maxDistance,referenceVector,maxAngularDistance);
    plane1 = select(ptCloud,inlierIndices);
    remainPtCloud = select(ptCloud,outlierIndices);
    
    % Set the region of interest to constrain the search for the second plane, left wall.
    roi = [-inf,inf;0.4,inf;-inf,inf];
    sampleIndices = findPointsInROI(remainPtCloud,roi);
    
    % Detect the second wall
    [model2,inlierIndices,outlierIndices] = pcfitplane(remainPtCloud,...
                maxDistance,'SampleIndices',sampleIndices);
    plane2 = select(remainPtCloud,inlierIndices);
    remainPtCloud = select(remainPtCloud,outlierIndices);
    
    % Detect the third wall
    [model3,inlierIndices,outlierIndices] = pcfitplane(remainPtCloud,...
                maxDistance,[1,0,0],60);
    plane3 = select(remainPtCloud,inlierIndices);
    remainPtCloud = select(remainPtCloud,outlierIndices);
    
    % model2.Normal: normal vector of left wall
    if abs(model3.Normal(2)) > abs(model2.Normal(2))
        a = model3.Normal;
        disp('plane 3')
        figure;pcshow(plane3);xlabel('X(m)');ylabel('Y(m)');zlabel('Z(m)')
        if a(2) > 0
            a = a.*[-1 -1 1];
        end
    else
        a = model2.Normal;
        disp('plane 2')
        figure;pcshow(plane2);xlabel('X(m)');ylabel('Y(m)');zlabel('Z(m)')
        if a(2) > 0
            a = a.*[-1 -1 1];
        end
    end

    b = [0,-1,0];
    angleme = acos(dot(a,b));
%     angle = atan2(norm(cross(a,b)),dot(a,b));
%     angle2 = atan2(norm(cross(a,b2)),dot(a,b2));
    if angleme < 0.0001% 0.0873  % 5 degree
        R = eye(3);   
    else
        test = cross(a, [0,-1, 0]);
        if test(3) > 0
            angleme = -angleme;
            disp('clockwise')
        else
            disp('counter-clockwise')
        end
        R = getRotationMatrix('z', angleme);
        R = R(1:3,1:3);
    end

    
    disp(angleme*180/pi)
