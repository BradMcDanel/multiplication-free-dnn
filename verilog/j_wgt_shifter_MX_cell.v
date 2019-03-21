module j_wgt_shifter_MX_cell #(
  parameter SRAM_DEPTH  = 256*256*4        ,
  parameter SHIFT_WIDTH = 8                ,
  parameter SRAM_ADDR_W = clog2(SRAM_DEPTH)
) (
  input                             clk          ,
  input                             reset_n      ,
  output wire [            8*1-1:0] sram_en      ,
  output wire [  SRAM_ADDR_W*8-1:0] sram_addr    ,
  input       [            8*8-1:0] sram_data    ,
  input                             shift_start  ,
  output                            shift_idle   ,
  input  wire [SRAM_ADDR_W*8*1-1:0] end_addr     ,
  input  wire [    SRAM_ADDR_W-1:0] img_size     ,
  output wire [SHIFT_WIDTH*8*1-1:0] serial_output,
  output wire [            8*1-1:0] serial_en
);
function [31:0] clog2 (input [31:0] x);
    reg [31:0] x_tmp;
    begin
            x_tmp = x-1;
            for(clog2=0; x_tmp>0; clog2=clog2+1) begin
                x_tmp = x_tmp >> 1;
            end
    end
endfunction

wire [          1*8- 1:0] mx_sram_en    ;
wire [SRAM_ADDR_W*8- 1:0] mx_sram_addr  ;
reg  [          1*8- 1:0] mx_shift_start;
wire [          1*8- 1:0] mx_shift_idle ;

always @(posedge clk) begin : proc_mx_shift_start
  if(~reset_n) begin
    mx_shift_start <= 8'b0;
  end else if(shift_start) begin
    mx_shift_start <= {(8){1'b1}};
  end else begin
    mx_shift_start <= 8'b0;
  end
end

genvar i;
generate
  for (i = 0; i < 8; i=i+1) begin: g_wgt_shifter
    j_wgt_shifter #(.SRAM_DEPTH(SRAM_DEPTH), .SHIFT_WIDTH(SHIFT_WIDTH)) i_j_wgt_shifter (
      .clk          (clk                                            ),
      .reset_n      (reset_n                                        ),
      .sram_en      (mx_sram_en       [i]                           ),
      .sram_addr    (mx_sram_addr     [i*SRAM_ADDR_W +: SRAM_ADDR_W]),
      .sram_data    (sram_data        [i*8           +: 8          ]),
      .shift_start  (mx_shift_start   [i]                           ),
      .shift_idle   (mx_shift_idle    [i]                           ),
      .end_addr     (end_addr         [i*SRAM_ADDR_W +: SRAM_ADDR_W]),
      .img_size     (img_size                                       ),
      .serial_output(serial_output    [i*SHIFT_WIDTH +: SHIFT_WIDTH]),
      .serial_en    (serial_en        [i]                           )
      //.serial_start (serial_start     [i]                           )
    );
  end
endgenerate

assign sram_en   = mx_sram_en;
assign sram_addr = mx_sram_addr;
assign shift_idle = &mx_shift_idle;


endmodule      