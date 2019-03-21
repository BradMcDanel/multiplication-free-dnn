
module hwgq #(
  parameter HEIGHT             = 2                     ,
  parameter START_QUANTIZE_BIT = 11                    ,
  parameter END_QUANTIZE_BIT   = START_QUANTIZE_BIT + 7
) (
  input                         clk             ,
  input                         requant_en      ,
  input                         start_quantizing,
  output reg [            31:0] reset             = 32'hffffffff,
  input      [  (4*HEIGHT-1):0] data_in         ,
  output     [8*(4*HEIGHT)-1:0] output_array
);

always @ (posedge clk) begin
  if(requant_en == 1'b1) begin
    begin : LOADING_RESET_SIGNAL
      integer n;
      for (n = 0; n < 31; n = n + 1)
        reset[n+1] <= reset[n];
    end
  end
  else begin  // when the start quantizing signal is 0
    begin : STOP_SIGNAL
      integer n;
      for (n = 0; n <= 31; n = n + 1)
        reset[n] <= 1;
    end
  end
end


always@(posedge clk) begin
  if(start_quantizing==1'b0) begin
    reset[0] <= 1;
  end
  else begin
    reset[0] <= 0;
  end
end

// then add the connection for the quantization
genvar i;
generate
  for (i=1; i<=HEIGHT*4; i=i+1) begin
    begin : QUANTIZATION_BLOCK
      activation_and_quantizer  #(.START_QUANTIZE_BIT(START_QUANTIZE_BIT), .END_QUANTIZE_BIT(END_QUANTIZE_BIT))
        pe (
          .clk(clk),
          .reset(reset[(8*((i-1)%4)+((i-1)/4))%32]),
          .data_in(data_in[i-1]),
          .output_array(output_array[8*i-1:8*(i-1)])
        );
    end
  end
endgenerate
endmodule 