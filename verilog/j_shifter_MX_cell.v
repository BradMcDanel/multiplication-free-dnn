module j_shifter_MX_cell #(
  parameter SRAM_DEPTH  = 256*256*4        ,
  parameter SRAM_ADDR_W = clog2(SRAM_DEPTH)
) (
  input                           clk            ,
  input                           reset_n        ,
  output wire                     sram_en        ,
  output wire [  SRAM_ADDR_W-1:0] sram_addr      ,
  input       [            1+7:0] sram_data      , // {skip, data[7:0]}
  input                           shift_start    ,
  output wire                     shift_start_o  ,
  output                          shift_idle     ,
  input       [          4*8-1:0] shift_ctrl     ,
  input  wire [SRAM_ADDR_W*8-1:0] start_addr     ,
  input  wire [  SRAM_ADDR_W-1:0] img_width_size ,
  input  wire [  SRAM_ADDR_W-1:0] img_height_size,
  output wire [              7:0] serial_output  ,
  output wire [              7:0] serial_en
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
  end else begin
    mx_shift_start <= {mx_shift_start[6:0], shift_start};
  end
end

assign shift_start_o = mx_shift_start[7];

genvar i;
generate
  for (i = 0; i < 8; i=i+1) begin: g_shifter
    j_shifter #(.SRAM_DEPTH(SRAM_DEPTH)) i_j_shifter (
      .clk            (clk                                            ),
      .reset_n        (reset_n                                        ),
      .sram_en        (mx_sram_en       [i]                           ),
      .sram_addr      (mx_sram_addr     [i*SRAM_ADDR_W +: SRAM_ADDR_W]),
      .sram_data      (sram_data                                      ),
      .shift_start    (mx_shift_start   [i]                           ),
      .shift_idle     (mx_shift_idle    [i]                           ),
      .shift_ctrl     (shift_ctrl       [i*4 +: 4]                    ),
      .start_addr     (start_addr       [i*SRAM_ADDR_W +: SRAM_ADDR_W]),
      .img_width_size (img_width_size                                 ),
      .img_height_size(img_height_size                                ),
      .serial_output  (serial_output    [i]                           ),
      .serial_en      (serial_en        [i]                           )
    );
  end
endgenerate

assign sram_en   = (|mx_sram_en);
assign sram_addr = {(SRAM_ADDR_W){mx_sram_en[0]}} & mx_sram_addr[0*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[1]}} & mx_sram_addr[1*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[2]}} & mx_sram_addr[2*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[3]}} & mx_sram_addr[3*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[4]}} & mx_sram_addr[4*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[5]}} & mx_sram_addr[5*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[6]}} & mx_sram_addr[6*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[7]}} & mx_sram_addr[7*SRAM_ADDR_W +: SRAM_ADDR_W];

assign shift_idle = &mx_shift_idle;

endmodule      
