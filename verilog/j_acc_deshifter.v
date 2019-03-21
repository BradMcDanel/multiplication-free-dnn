module j_acc_deshifter #(
  parameter SRAM_DEPTH  = 256*256*4        ,
  parameter SRAM_ADDR_W = clog2(SRAM_DEPTH)
) (
  input                         clk         ,
  input                         reset_n     ,
  output reg                    sram_en     ,
  output reg  [SRAM_ADDR_W-1:0] sram_addr   ,
  output wire [           31:0] sram_data   ,
  input                         shift_start ,
  output                        shift_idle  ,
  input  wire [SRAM_ADDR_W-1:0] start_addr  ,
  input  wire [SRAM_ADDR_W-1:0] img_size    ,
  input  wire                   serial_input,
  input  wire                   serial_en
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

  parameter S_IDEL              = 0;
  parameter S_SHIFT             = 1;

  reg                    fsm_state     ;
  reg                    fsm_state_nxt ;
  reg  [SRAM_ADDR_W-1:0] cur_cnt       ;
  wire [SRAM_ADDR_W-1:0] cur_cnt_nxt   ;
  wire                   cur_cnt_inc   ;
  wire                   cur_cnt_clr   ;
  reg  [            4:0] serial_cnt    ;
  wire                   serial_cnt_inc;
  wire                   serial_end    ;
  reg                    serial_en_dly ;

  always @(posedge clk) begin : proc_serial_en_dly
    if(~reset_n) begin
      serial_en_dly <= 0;
    end else begin
      serial_en_dly <= serial_en;
    end
  end

  assign serial_cnt_inc = serial_en;
  assign serial_end     = serial_cnt == 5'b11111;
  reg    serial_end_dly;

  always @(posedge clk) begin : proc_serial_end_dly
    if(~reset_n) begin
      serial_end_dly <= 0;
    end else begin
      serial_end_dly <= serial_end;
    end
  end

  always @(posedge clk) begin : proc_serial_cnt
    if(~reset_n) begin
      serial_cnt <= 0;
    end else if(serial_cnt_inc) begin
      serial_cnt <= serial_cnt + 1;
    end
  end


  always @(posedge clk) begin : proc_fsm_state
    if(~reset_n) begin
      fsm_state <= S_IDEL;
    end else begin
      fsm_state <= fsm_state_nxt;
    end
  end

  always @(*) begin : proc_fsm_state_nxt
    fsm_state_nxt = fsm_state;
    case (fsm_state)
      S_IDEL: begin
        if(shift_start)
          fsm_state_nxt = S_SHIFT;
      end
      S_SHIFT: begin
        if(cur_cnt_clr) begin
          fsm_state_nxt = S_IDEL;
        end
      end
    endcase
  end

  assign shift_idle = fsm_state == S_IDEL;

  assign cur_cnt_inc = serial_en & serial_end;
  assign cur_cnt_clr = (cur_cnt == img_size) & cur_cnt_inc;
  assign cur_cnt_nxt = cur_cnt+1;

  always @(posedge clk) begin : proc_cur_cnt
    if(~reset_n) begin
      cur_cnt <= 0;
    end else if(cur_cnt_clr) begin
      cur_cnt <= 0;
    end else if(cur_cnt_inc) begin
      cur_cnt <= cur_cnt_nxt;
    end
  end

  reg  [31:0] deserial_reg    ;
  wire [31:0] deserial_reg_nxt;
  assign deserial_reg_nxt = {serial_input, deserial_reg[31:1]};

  always @(posedge clk) begin : proc_deserial_reg
    if(~reset_n) begin
      deserial_reg <= 0;
    end else if(serial_en) begin
      deserial_reg <= deserial_reg_nxt;
    end
  end

  assign sram_data = deserial_reg;
 
  // always @(posedge clk) begin : proc_sram_en
  //   if(~reset_n) begin
  //     sram_en <= 0;
  //   end else begin
  //     sram_en <= serial_en_dly & serial_end_dly;
  //   end
  // end

  always @(*) begin : proc_sram_en
    sram_en = serial_en_dly & serial_end_dly;
  end

  reg [SRAM_ADDR_W-1:0] sram_addr_nxt;

  always @(*) begin : proc_sram_addr_nxt
    sram_addr_nxt = sram_addr;
    if((fsm_state == S_IDEL) & shift_start)
      sram_addr_nxt = start_addr;
    else if(serial_en_dly & serial_end_dly)
      sram_addr_nxt = sram_addr + 4;
    else
      sram_addr_nxt = sram_addr;
  end

  always @(posedge clk) begin : proc_sram_addr
    if(~reset_n) begin
      sram_addr <= 0;
    end else begin
      sram_addr <= sram_addr_nxt;
    end
  end

endmodule      