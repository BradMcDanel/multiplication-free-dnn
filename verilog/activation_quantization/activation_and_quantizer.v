module activation_and_quantizer #(parameter START_QUANTIZE_BIT = 11, END_QUANTIZE_BIT = 18) (
  input            clk         ,
  input            reset       ,
  input            data_in     ,
  output reg [4:0] counter       = 5'b0  ,
  output reg [7:0] output_array
);

  wire [7:0] reg_array_wire;
  wire       dummy_reg_wire;
  reg [7:0] reg_array = 8'b0;
  reg       dummy_reg = 1'b0;
  reg       flag      = 1'b0;

  // connect to the demux
  demux1to9 #(.CATCH_START_BIT(START_QUANTIZE_BIT-2)) tt1 (
    .clk       (clk              ),
    .sel       (counter          ),
    .Data_in   (data_in          ),
    .Data_out_0(reg_array_wire[0]),
    .Data_out_1(reg_array_wire[1]),
    .Data_out_2(reg_array_wire[2]),
    .Data_out_3(reg_array_wire[3]),
    .Data_out_4(reg_array_wire[4]),
    .Data_out_5(reg_array_wire[5]),
    .Data_out_6(reg_array_wire[6]),
    .Data_out_7(reg_array_wire[7]),
    .Data_out_8(dummy_reg_wire   )
  );

  always@(posedge clk) begin
    if (reset) begin
      counter <= 5'b0;
      output_array <= 8'b0;
    end
    else begin
      if(counter > END_QUANTIZE_BIT) begin
        if(dummy_reg == 1) begin
          flag <= 1;
        end
      end
      if(counter == 5'b00000) begin
        if(dummy_reg == 1'b0) begin
          if(flag == 1'b1) begin    // if flag1 is 1 and 32 bits is 0, which means the value is clipped to 6
            output_array <= 8'b11111111;
            flag         <= 0;
          end
          else begin
            output_array <= reg_array;
            flag         <= 0;
          end
        end
        else begin   // if the value is negative, clip the value to 0
          output_array <= 8'b0;
          flag         <= 0;
        end
      end
      counter   <= counter + 1;
      // load the dummy regs
      dummy_reg <= dummy_reg_wire;
      // load the reg_array
      reg_array <= reg_array_wire;
    end
  end 

endmodule
