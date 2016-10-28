function d = for_test_BP(x)

 x1 = x(1);
 x2 = x(2);
 x3 = x(3);
 
 if x1<0.2 && x2>0.8 && x3>0.7
     d1 = 1; d2 = 0; d3 = 0;
 elseif x1>0.5 && x2<0.5 && x3<0.4
     d1 = 0; d2 = 1; d3 = 0;
 else
     d1 = 0; d2 = 0; d3 = 1;
 end
 
 d = [d1;d2;d3];
 
end