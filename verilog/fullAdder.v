module fullAdder(Sum,Cout,Cin,In1,In2);
   input wire In1,In2,Cin;
   output wire Sum,Cout;
   assign Cout = (In1&In2)|(In1&Cin)|(In2&Cin);
   assign Sum = In1^In2^Cin;
   //assign {Cout,Sum} = In1+In2+Cin;
endmodule
