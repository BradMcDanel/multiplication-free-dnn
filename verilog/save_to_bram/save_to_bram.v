//--------------------------------------------
// Note: This module takes 8 systolic cell as input, and saves the 4*8 = 32 outputs to the bram
// First input comes one cycle later after setting saving_mode = 1
// When saving_mode = 0, user can load the content in the memory
module save_to_bram #(
  parameter SRAM_DEPTH   = 256*256*4        ,
  parameter SRAM_ADDR_W  = 18               ,
  parameter IMAGE_WIDTH  = 5                ,
  parameter IMAGE_HEIGHT = 5
) (
  input                           clk            ,
  input                           saving_mode    , // saving_mode = 0 if user want to accress the sram
  input       [      (8*4*8-1):0] data_in        , // is the data output of the requantization block
  input  wire [             17:0] user_input_addr, // is the address that user want to access
  input  wire [SRAM_ADDR_W*8-1:0] starting_addr  , // is the starting addr of each of 8 channels
  output reg  [              7:0] sram_in        , // is the user access bram output
  output reg                      sram_we        ,
  output reg                      sram_en        ,
  //output reg  [    SRAM_ADDR_W-1:0] sram_addr   ,
  output wire [             17:0] sram_addr_ff
);

//function [31:0] clog2 (input [31:0] x);
//    reg [31:0] x_tmp;
//    begin
//            x_tmp = x-1;
//            for(clog2=0; x_tmp>0; clog2=clog2+1) begin
//                x_tmp = x_tmp >> 1;
//            end
//    end
//endfunction

// 
  reg [           15:0] total_cycles_of_operations = 0           ;
  reg [            4:0] counter                    = 0           ;
  reg                   saving_mode_reg            = 0           ;
  reg [SRAM_ADDR_W-1:0] sram_addr                  = 0           ;
  reg [            8:0] logger0                    = 9'b0        ;
  reg [            8:0] logger1                    = 9'b0        ;
  reg [            8:0] logger2                    = 9'b0        ;
  reg [            8:0] logger3                    = 9'b0        ;
  reg [            8:0] logger4                    = 9'b0        ;
  reg [            8:0] logger5                    = 9'b0        ;
  reg [            8:0] logger6                    = 9'b0        ;
  reg [            8:0] logger7                    = 9'b111111111;

  assign sram_addr_ff[0]  = (saving_mode_reg&sram_addr[0])|(~saving_mode_reg&user_input_addr[0]);
  assign sram_addr_ff[1]  = (saving_mode_reg&sram_addr[1])|(~saving_mode_reg&user_input_addr[1]);
  assign sram_addr_ff[2]  = (saving_mode_reg&sram_addr[2])|(~saving_mode_reg&user_input_addr[2]);
  assign sram_addr_ff[3]  = (saving_mode_reg&sram_addr[3])|(~saving_mode_reg&user_input_addr[3]);
  assign sram_addr_ff[4]  = (saving_mode_reg&sram_addr[4])|(~saving_mode_reg&user_input_addr[4]);
  assign sram_addr_ff[5]  = (saving_mode_reg&sram_addr[5])|(~saving_mode_reg&user_input_addr[5]);
  assign sram_addr_ff[6]  = (saving_mode_reg&sram_addr[6])|(~saving_mode_reg&user_input_addr[6]);
  assign sram_addr_ff[7]  = (saving_mode_reg&sram_addr[7])|(~saving_mode_reg&user_input_addr[7]);
  assign sram_addr_ff[8]  = (saving_mode_reg&sram_addr[8])|(~saving_mode_reg&user_input_addr[8]);
  assign sram_addr_ff[9]  = (saving_mode_reg&sram_addr[9])|(~saving_mode_reg&user_input_addr[9]);
  assign sram_addr_ff[10] = (saving_mode_reg&sram_addr[10])|(~saving_mode_reg&user_input_addr[10]);
  assign sram_addr_ff[11] = (saving_mode_reg&sram_addr[11])|(~saving_mode_reg&user_input_addr[11]);
  assign sram_addr_ff[12] = (saving_mode_reg&sram_addr[12])|(~saving_mode_reg&user_input_addr[12]);
  assign sram_addr_ff[13] = (saving_mode_reg&sram_addr[13])|(~saving_mode_reg&user_input_addr[13]);
  assign sram_addr_ff[14] = (saving_mode_reg&sram_addr[14])|(~saving_mode_reg&user_input_addr[14]);
  assign sram_addr_ff[15] = (saving_mode_reg&sram_addr[15])|(~saving_mode_reg&user_input_addr[15]);
  assign sram_addr_ff[16] = (saving_mode_reg&sram_addr[16])|(~saving_mode_reg&user_input_addr[16]);
  assign sram_addr_ff[17] = (saving_mode_reg&sram_addr[17])|(~saving_mode_reg&user_input_addr[17]);
// connect to the sram
// sram ram0 (.clk(clk), .we(sram_we), .en(sram_en), .addr(sram_addr), .data_i(sram_in), .data_o());

  always @(posedge clk) begin
    if(saving_mode == 1'b1) begin
      total_cycles_of_operations <= total_cycles_of_operations + 1;
    end
  end

  always @ (posedge clk) begin
    if(saving_mode == 1'b0) begin
      logger0                    <= 0;
      logger1                    <= 0;
      logger2                    <= 0;
      logger3                    <= 0;
      logger4                    <= 0;
      logger5                    <= 0;
      logger6                    <= 0;
      logger7                    <= 9'b111111111;
      counter                    <= 0;
      sram_addr                  <= 0;
      sram_en                    <= 1;
      sram_we                    <= 0;
      sram_in                    <= 0;
      total_cycles_of_operations <= 0;
    end
    else begin
      if (total_cycles_of_operations >= 8*(IMAGE_WIDTH*IMAGE_HEIGHT-1) + 9 ) begin
        sram_en <= 1'b0;
        sram_we <= 1'b0;
      end
      if (total_cycles_of_operations < 8*(IMAGE_WIDTH*IMAGE_HEIGHT-1) + 9 ) begin
        sram_en <= 1'b1;
        sram_we <= 1'b1;
      end
      if(counter == 1) begin
        sram_addr    <= starting_addr[SRAM_ADDR_W-1:0] + logger0;
        logger0      <= logger0 + 1;
        sram_in[7:0] <= data_in[7:0];
      end
      if(counter == 2) begin
        sram_addr    <= starting_addr[2*SRAM_ADDR_W-1:SRAM_ADDR_W] + logger1;
        logger1      <= logger1 + 1;
        sram_in[7:0] <= data_in[39:32];
      end
      if(counter == 3) begin
        sram_addr    <= starting_addr[3*SRAM_ADDR_W-1:2*SRAM_ADDR_W] + logger2;
        logger2      <= logger2 + 1;
        sram_in[7:0] <= data_in[71:64];
      end
      if(counter == 4) begin
        sram_addr    <= starting_addr[4*SRAM_ADDR_W-1:3*SRAM_ADDR_W] + logger3;
        logger3      <= logger3 + 1;
        sram_in[7:0] <= data_in[103:96];
      end
      if(counter == 5) begin
        sram_addr    <= starting_addr[5*SRAM_ADDR_W-1:4*SRAM_ADDR_W] + logger4;
        logger4      <= logger4 + 1;
        sram_in[7:0] <= data_in[135:128];
      end
      if(counter == 6) begin
        sram_addr    <= starting_addr[6*SRAM_ADDR_W-1:5*SRAM_ADDR_W] + logger5;
        logger5      <= logger5 + 1;
        sram_in[7:0] <= data_in[167:160];
      end
      if(counter == 7) begin
        sram_addr    <= starting_addr[7*SRAM_ADDR_W-1:6*SRAM_ADDR_W] + logger6;
        logger6      <= logger6 + 1;
        sram_in[7:0] <= data_in[199:192];
      end
      if(counter == 8) begin
        sram_addr    <= starting_addr[8*SRAM_ADDR_W-1:7*SRAM_ADDR_W] + logger7;
        logger7      <= logger7 + 1;
        sram_in[7:0] <= data_in[231:224];
      end
      if(counter == 9) begin
        sram_addr    <= starting_addr[SRAM_ADDR_W-1:0] + logger0;
        logger0      <= logger0 + 1;
        sram_in[7:0] <= data_in[15:8];
      end
      if(counter == 10) begin
        sram_addr    <= starting_addr[2*SRAM_ADDR_W-1:SRAM_ADDR_W] + logger1;
        logger1      <= logger1 + 1;
        sram_in[7:0] <= data_in[47:40];
      end
      if(counter == 11) begin
        sram_addr    <= starting_addr[3*SRAM_ADDR_W-1:2*SRAM_ADDR_W] + logger2;
        logger2      <= logger2 + 1;
        sram_in[7:0] <= data_in[79:72];
      end
      if(counter == 12) begin
        sram_addr    <= starting_addr[4*SRAM_ADDR_W-1:3*SRAM_ADDR_W] + logger3;
        logger3      <= logger3 + 1;
        sram_in[7:0] <= data_in[111:104];
      end
      if(counter == 13) begin
        sram_addr    <= starting_addr[5*SRAM_ADDR_W-1:4*SRAM_ADDR_W] + logger4;
        logger4      <= logger4 + 1;
        sram_in[7:0] <= data_in[143:136];
      end
      if(counter == 14) begin
        sram_addr    <= starting_addr[6*SRAM_ADDR_W-1:5*SRAM_ADDR_W] + logger5;
        logger5      <= logger5 + 1;
        sram_in[7:0] <= data_in[175:168];
      end
      if(counter == 15) begin
        sram_addr    <= starting_addr[7*SRAM_ADDR_W-1:6*SRAM_ADDR_W] + logger6;
        logger6      <= logger6 + 1;
        sram_in[7:0] <= data_in[207:200];
      end
      if(counter == 16) begin
        sram_addr    <= starting_addr[8*SRAM_ADDR_W-1:7*SRAM_ADDR_W] + logger7;
        logger7      <= logger7 + 1;
        sram_in[7:0] <= data_in[239:232];
      end
      if(counter == 17) begin
        sram_addr    <= starting_addr[SRAM_ADDR_W-1:0] + logger0;
        logger0      <= logger0 + 1;
        sram_in[7:0] <= data_in[23:16];
      end
      if(counter == 18) begin
        sram_addr    <= starting_addr[2*SRAM_ADDR_W-1:SRAM_ADDR_W] + logger1;
        logger1      <= logger1 + 1;
        sram_in[7:0] <= data_in[55:48];
      end
      if(counter == 19) begin
        sram_addr    <= starting_addr[3*SRAM_ADDR_W-1:2*SRAM_ADDR_W] + logger2;
        logger2      <= logger2 + 1;
        sram_in[7:0] <= data_in[87:80];
      end
      if(counter == 20) begin
        sram_addr    <= starting_addr[4*SRAM_ADDR_W-1:3*SRAM_ADDR_W] + logger3;
        logger3      <= logger3 + 1;
        sram_in[7:0] <= data_in[119:112];
      end
      if(counter == 21) begin
        sram_addr    <= starting_addr[5*SRAM_ADDR_W-1:4*SRAM_ADDR_W] + logger4;
        logger4      <= logger4 + 1;
        sram_in[7:0] <= data_in[151:144];
      end
      if(counter == 22) begin
        sram_addr    <= starting_addr[6*SRAM_ADDR_W-1:5*SRAM_ADDR_W] + logger5;
        logger5      <= logger5 + 1;
        sram_in[7:0] <= data_in[183:176];
      end
      if(counter == 23) begin
        sram_addr    <= starting_addr[7*SRAM_ADDR_W-1:6*SRAM_ADDR_W] + logger6;
        logger6      <= logger6 + 1;
        sram_in[7:0] <= data_in[215:208];
      end
      if(counter == 24) begin
        sram_addr    <= starting_addr[8*SRAM_ADDR_W-1:7*SRAM_ADDR_W] + logger7;
        logger7      <= logger7 + 1;
        sram_in[7:0] <= data_in[247:240];
      end
      if(counter == 25) begin
        sram_addr    <= starting_addr[SRAM_ADDR_W-1:0] + logger0;
        logger0      <= logger0 + 1;
        sram_in[7:0] <= data_in[31:24];
      end
      if(counter == 26) begin
        sram_addr    <= starting_addr[2*SRAM_ADDR_W-1:SRAM_ADDR_W] + logger1;
        logger1      <= logger1 + 1;
        sram_in[7:0] <= data_in[63:56];
      end
      if(counter == 27) begin
        sram_addr    <= starting_addr[3*SRAM_ADDR_W-1:2*SRAM_ADDR_W] + logger2;
        logger2      <= logger2 + 1;
        sram_in[7:0] <= data_in[95:88];
      end
      if(counter == 28) begin
        sram_addr    <= starting_addr[4*SRAM_ADDR_W-1:3*SRAM_ADDR_W] + logger3;
        logger3      <= logger3 + 1;
        sram_in[7:0] <= data_in[127:120];
      end
      if(counter == 29) begin
        sram_addr    <= starting_addr[5*SRAM_ADDR_W-1:4*SRAM_ADDR_W] + logger4;
        logger4      <= logger4 + 1;
        sram_in[7:0] <= data_in[159:152];
      end
      if(counter == 30) begin
        sram_addr    <= starting_addr[6*SRAM_ADDR_W-1:5*SRAM_ADDR_W] + logger5;
        logger5      <= logger5 + 1;
        sram_in[7:0] <= data_in[191:184];
      end
      if(counter == 31) begin
        sram_addr    <= starting_addr[7*SRAM_ADDR_W-1:6*SRAM_ADDR_W] + logger6;
        logger6      <= logger6 + 1;
        sram_in[7:0] <= data_in[223:216];
      end
      if(counter == 0) begin
        sram_addr    <= starting_addr[8*SRAM_ADDR_W-1:7*SRAM_ADDR_W] + logger7;
        logger7      <= logger7 + 1;
        sram_in[7:0] <= data_in[255:248];
      end
      counter <= counter+1;
    end
    saving_mode_reg <= saving_mode;
  end
  endmodule 
