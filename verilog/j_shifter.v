module j_shifter #(
  parameter SRAM_DEPTH  = 256*256*4        ,
  parameter SRAM_ADDR_W = clog2(SRAM_DEPTH)
) (
  input                         clk            ,
  input                         reset_n        ,
  output reg                    sram_en        ,
  output reg  [SRAM_ADDR_W-1:0] sram_addr      ,
  input       [          1+7:0] sram_data      , // {skip, data[7:0]}
  input                         shift_start    ,
  output                        shift_idle     ,
  input       [            3:0] shift_ctrl     ,
  input  wire [SRAM_ADDR_W-1:0] start_addr     ,
  input  wire [SRAM_ADDR_W-1:0] img_width_size ,
  input  wire [SRAM_ADDR_W-1:0] img_height_size,
  output wire                   serial_output  ,
  output wire                   serial_en
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
  parameter S_LOAD_DATA         = 1;
  parameter S_SHIFT             = 2;

  parameter SHT_KEEP       = 0;
  parameter SHT_UP         = 1;
  parameter SHT_DOWN       = 2;
  parameter SHT_LEFT       = 3;
  parameter SHT_RIGHT      = 4;
  parameter SHT_RIGHT_UP   = 5;
  parameter SHT_RIGHT_DOWN = 6;
  parameter SHT_LEFT_DOWN  = 7;
  parameter SHT_LEFT_UP    = 8;

  reg  [            2:0] fsm_state       ;
  reg  [            2:0] fsm_state_nxt   ;
  reg  [            2:0] shift_cnt       ;
  wire                   shift_cnt_inc   ;
  wire                   shift_cnt_done  ;
  reg  [SRAM_ADDR_W-1:0] cur_w_cnt       ;
  reg  [SRAM_ADDR_W-1:0] cur_h_cnt       ;
  wire [SRAM_ADDR_W-1:0] cur_w_cnt_nxt   ;
  wire [SRAM_ADDR_W-1:0] cur_h_cnt_nxt   ;
  wire                   cur_w_cnt_inc   ;
  wire                   cur_w_cnt_clr   ;
  wire                   cur_h_cnt_inc   ;
  wire                   cur_h_cnt_clr   ;
  reg                    fake_sram_en, fake_sram_en_dly;
  wire                   fake_sram_en_nxt;
  wire                   img_cnt_inc     ;

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
      S_LOAD_DATA: begin
        fsm_state_nxt = S_SHIFT;
      end
      S_SHIFT: begin
        if(cur_w_cnt_clr & cur_h_cnt_clr) begin
          fsm_state_nxt = S_IDEL;
        end else if(shift_cnt_done) begin
          fsm_state_nxt = S_LOAD_DATA;
        end
      end
      default : /* default */;
    endcase
  end

  assign cur_w_cnt_inc = img_cnt_inc;
  assign cur_w_cnt_clr = (cur_w_cnt == img_width_size) & img_cnt_inc;
  assign cur_h_cnt_inc = cur_w_cnt_clr;
  assign cur_h_cnt_clr = (cur_h_cnt == img_height_size) & cur_w_cnt_clr;

  assign cur_w_cnt_nxt = cur_w_cnt+1;
  assign cur_h_cnt_nxt = cur_h_cnt+1;

  always @(posedge clk) begin : proc_cur_w_cnt
    if(~reset_n) begin
      cur_w_cnt <= 0;
    end else if(cur_w_cnt_clr) begin
      cur_w_cnt <= 0;
    end else if(cur_w_cnt_inc) begin
      cur_w_cnt <= cur_w_cnt_nxt;
    end
  end

  always @(posedge clk) begin : proc_cur_h_cnt
    if(~reset_n) begin
      cur_h_cnt <= 0;
    end else if(cur_h_cnt_clr) begin
      cur_h_cnt <= 0;
    end else if(cur_h_cnt_inc) begin
      cur_h_cnt <= cur_h_cnt_nxt;
    end
  end  


  // Shift control
  reg shift_state_dly;
  assign shift_cnt_done = shift_cnt == 3'b101;
  assign shift_cnt_inc  = ~(shift_cnt == 3'b0) | shift_state_dly;
  assign img_cnt_inc    = shift_cnt == 3'b101;
  always @(posedge clk) begin : proc_shift_state_dly
    if(~reset_n) begin
      shift_state_dly <= 0;
    end else begin
      shift_state_dly <= (fsm_state == S_SHIFT);
    end
  end

  wire [2:0] shift_cnt_nxt = shift_cnt + 1;
  always @(posedge clk) begin : proc_shift_cnt
    if(~reset_n) begin
      shift_cnt <= 0;
    end else if(shift_cnt_inc) begin
      shift_cnt <= shift_cnt_nxt;
    end
  end

  assign fake_sram_en_nxt = (fsm_state == S_LOAD_DATA) | shift_start;
  always @(posedge clk) begin : proc_fake_sram_en
    if(~reset_n) begin
      {fake_sram_en_dly, fake_sram_en} <= 0;
    end else begin
      {fake_sram_en_dly, fake_sram_en} <= {fake_sram_en, fake_sram_en_nxt};
    end
  end

  reg sram_en_dly;
  wire sram_en_dly_nxt = sram_en;
  always @(posedge clk) begin : proc_sram_en_dly
    if(~reset_n) begin
      sram_en_dly <= 0;
    end else begin
      sram_en_dly <= sram_en_dly_nxt;
    end
  end

  reg shift_cnt_inc_dly0,shift_cnt_inc_dly1;
  always @(posedge clk) begin : proc_shift_cnt_inc_dly
    if(~reset_n) begin
      {shift_cnt_inc_dly1, shift_cnt_inc_dly0} <= 0;
    end else begin
      {shift_cnt_inc_dly1, shift_cnt_inc_dly0} <= {shift_cnt_inc_dly0, shift_cnt_inc};
    end
  end

  reg  [7:0] sram_data_latch  ;
  wire       zero_skip        ;
  reg        zero_skip_tmp    ;
  wire       zero_skip_tmp_en ;
  reg        zero_skip_tmp_clr;
  reg        zero_skip_tmp_clr_dly1;
  reg        zero_skip_tmp_clr_dly2;
  wire       zero_skip_tmp_nxt;

  always @(posedge clk) begin : proc_zero_skip_tmp_clr_dly
    if(~reset_n) begin
      {zero_skip_tmp_clr_dly2, zero_skip_tmp_clr_dly1} <= 0;
    end else begin
      {zero_skip_tmp_clr_dly2, zero_skip_tmp_clr_dly1} <= {zero_skip_tmp_clr_dly1, zero_skip_tmp_clr};
    end
  end


  assign zero_skip = 
        zero_skip_tmp & fake_sram_en_dly ? 1'b1                                                       :
        shift_ctrl == SHT_KEEP           ? 1'b0                                                       :
        shift_ctrl == SHT_UP             ? (cur_h_cnt==0                                             ):
        shift_ctrl == SHT_DOWN           ? (cur_h_cnt==img_height_size                               ):
        shift_ctrl == SHT_LEFT           ? (cur_w_cnt==0                                             ):
        shift_ctrl == SHT_RIGHT          ? (cur_w_cnt==img_width_size                                ):
        shift_ctrl == SHT_RIGHT_UP       ? (cur_h_cnt==0) | (cur_w_cnt==img_width_size               ):
        shift_ctrl == SHT_RIGHT_DOWN     ? (cur_h_cnt==img_height_size) | (cur_w_cnt==img_width_size ):
        shift_ctrl == SHT_LEFT_DOWN      ? (cur_h_cnt==img_height_size) | (cur_w_cnt==0              ):
        shift_ctrl == SHT_LEFT_UP        ? (cur_h_cnt==0) | (cur_w_cnt==0                            ):
        1'b0;

  always @(posedge clk) begin : proc_sram_data_latch
    if(~reset_n) begin
      sram_data_latch <= 0;
    end else if(fake_sram_en_dly) begin
      sram_data_latch <= zero_skip ? 8'b0: sram_data[7:0];
    end else if(shift_cnt_inc_dly0) begin
      sram_data_latch <= {1'b0, sram_data_latch[7:1]};
    end
  end

  assign zero_skip_tmp_en = sram_en_dly & sram_data[8] | zero_skip_tmp_clr_dly2;
  assign zero_skip_tmp_nxt = zero_skip_tmp_clr_dly2 ? 1'b0 : 
                             sram_en_dly ? sram_data[8]    : zero_skip_tmp;

  always @(posedge clk) begin : proc_zero_skip_tmp
    if(~reset_n) begin
      zero_skip_tmp <= 0;
    end else if(zero_skip_tmp_en) begin
      zero_skip_tmp <= zero_skip_tmp_nxt;
    end
  end

  reg sram_en_nxt;
  always @(posedge clk) begin : proc_sram_en
    if(~reset_n) begin
      sram_en <= 0;
    end else begin
      sram_en <= sram_en_nxt;
    end
  end

  reg [SRAM_ADDR_W-1:0] sram_addr_nxt;
 
  always @(*) begin : proc_sram_addr_nxt
      sram_addr_nxt     = sram_addr;
      sram_en_nxt       = 1'b0;
      zero_skip_tmp_clr = 1'b0;
      if((shift_start == 1'b1) & (fsm_state == S_IDEL)) begin
        zero_skip_tmp_clr = 1'b1;
        case(shift_ctrl)
          SHT_KEEP       : begin
            // latch first point
            sram_addr_nxt = start_addr;
            sram_en_nxt   = 1'b1;
          end
          SHT_UP         : begin
            // latch first point
            sram_addr_nxt = start_addr + img_width_size;
            sram_en_nxt   = 1'b1;
          end
          SHT_DOWN       : begin
            // latch first point
            sram_addr_nxt = start_addr + img_width_size + 1;
            sram_en_nxt   = 1'b1;
          end
          SHT_LEFT       : begin
            // skip first point
            sram_addr_nxt = start_addr;
            sram_en_nxt   = 1'b0;
          end
          SHT_RIGHT      : begin
            sram_addr_nxt = start_addr + 1;
            sram_en_nxt   = 1'b1;
          end
          SHT_RIGHT_UP   : begin
            sram_addr_nxt = start_addr + 1;
            sram_en_nxt   = 1'b0;
          end
          SHT_RIGHT_DOWN : begin
            sram_addr_nxt = start_addr + img_width_size + 2;
            sram_en_nxt   = 1'b1;
          end
          SHT_LEFT_UP    : begin
            sram_addr_nxt = start_addr;
            sram_en_nxt   = 1'b0;
          end
          SHT_LEFT_DOWN  : begin
            sram_addr_nxt = start_addr + img_width_size + 1;
            sram_en_nxt   = 1'b0;
          end
        endcase // shift_ctrl
      end else if(fsm_state == S_LOAD_DATA) begin
        case(shift_ctrl)
          SHT_KEEP       : begin
            sram_addr_nxt = sram_addr + 1;
            sram_en_nxt   = zero_skip_tmp ? 1'b0 : 1'b1;
            zero_skip_tmp_clr = zero_skip_tmp;
          end
          SHT_UP         : begin
            sram_addr_nxt = (cur_h_cnt == 0              ) ? sram_addr : 
                                                             sram_addr + 1;
            sram_en_nxt   = (cur_h_cnt == 0              ) ? 1'b0 :
                            zero_skip_tmp                  ? 1'b0 : 1'b1;
            zero_skip_tmp_clr = zero_skip_tmp;
          end
          SHT_DOWN       : begin
            sram_addr_nxt = sram_addr + 1;
            sram_en_nxt   = (cur_h_cnt == img_height_size) ? 1'b0 :
                            zero_skip_tmp                  ? 1'b0 : 1'b1;
            zero_skip_tmp_clr = zero_skip_tmp;
          end
          SHT_LEFT       : begin
            sram_addr_nxt = (cur_h_cnt == 0) & (cur_w_cnt == 0) ?  sram_addr      :
                            (cur_w_cnt == 1)                    ?  sram_addr      :
                            // (cur_w_cnt == 0) & zero_skip_tmp    ? (sram_addr + 1) :
                            // (cur_w_cnt == 0) &~zero_skip_tmp    ? (sram_addr + 2) :
                            sram_addr + 1;
            sram_en_nxt   = (cur_w_cnt == 0) ? 1'b0 :
                                         zero_skip_tmp    ? 1'b0 : 1'b1;
            zero_skip_tmp_clr = zero_skip_tmp;
          end
          SHT_RIGHT      : begin
            sram_addr_nxt = //(cur_h_cnt == 0) & (cur_w_cnt == img_width_size) ?  sram_addr    :
                                               (cur_w_cnt == img_width_size) ?  sram_addr + 1:
                            //(cur_w_cnt == 0) & zero_skip_tmp    ? (sram_addr + 1) :
                            //(cur_w_cnt == 0) &~zero_skip_tmp    ? (sram_addr + 2) :
                            sram_addr + 1;
            sram_en_nxt   = (cur_w_cnt == img_width_size) ? 1'b0 :
                                         zero_skip_tmp    ? 1'b0 : 1'b1;
            zero_skip_tmp_clr = zero_skip_tmp;
          end
          SHT_RIGHT_UP   : begin
            sram_addr_nxt = (cur_h_cnt == 0)                                  ?  sram_addr    :
                            (cur_w_cnt == 0)              & (cur_h_cnt == 1)  ?  sram_addr    :
                            //(cur_w_cnt == 0) & zero_skip_tmp    ? (sram_addr + 1) :
                            //(cur_w_cnt == 0) &~zero_skip_tmp    ? (sram_addr + 2) :
                            sram_addr + 1;
            sram_en_nxt   = (cur_w_cnt == img_width_size) || (cur_h_cnt == 0) ? 1'b0 :
                                                            zero_skip_tmp     ? 1'b0 : 1'b1;
            zero_skip_tmp_clr = zero_skip_tmp;
          end
          SHT_RIGHT_DOWN : begin
            sram_addr_nxt = (cur_h_cnt == img_height_size)                    ?  sram_addr    :
                            //(cur_w_cnt == 0)              & (cur_h_cnt == 1)  ?  sram_addr    :
                            //(cur_w_cnt == 0) & zero_skip_tmp    ? (sram_addr + 1) :
                            //(cur_w_cnt == 0) &~zero_skip_tmp    ? (sram_addr + 2) :
                            sram_addr + 1;
            sram_en_nxt   = (cur_w_cnt == img_width_size) || (cur_h_cnt == img_height_size) ? 1'b0 :
                                                                          zero_skip_tmp     ? 1'b0 : 1'b1;
            zero_skip_tmp_clr = zero_skip_tmp;
          end
          SHT_LEFT_UP    : begin
            sram_addr_nxt = (cur_h_cnt == 0)                      ? sram_addr    :
                            (cur_w_cnt == 0) && (cur_h_cnt == 1)  ? sram_addr    :
                            (cur_w_cnt == 1) && (cur_h_cnt == 1)  ? sram_addr    :
                            (cur_w_cnt == 0) && (cur_h_cnt != 0)  ? sram_addr+1  :
                            //(cur_w_cnt == 0)              & (cur_h_cnt == 1)  ?  sram_addr    :
                            //(cur_w_cnt == 0) & zero_skip_tmp    ? (sram_addr + 1) :
                            //(cur_w_cnt == 0) &~zero_skip_tmp    ? (sram_addr + 2) :
                            sram_addr + 1;
            sram_en_nxt   = (cur_w_cnt == 0) || (cur_h_cnt == 0) ? 1'b0 :
                                               zero_skip_tmp     ? 1'b0 : 1'b1;
            zero_skip_tmp_clr = zero_skip_tmp;
          end
          SHT_LEFT_DOWN  : begin
            sram_addr_nxt = //(cur_h_cnt == 0)                      ? sram_addr    :
                            //(cur_w_cnt == 0)                        ? sram_addr    :
                            (cur_w_cnt == 1) && (cur_h_cnt == 0)  ? sram_addr    :
                            //(cur_w_cnt == 0) && (cur_h_cnt != 0)  ? sram_addr+1  :
                            //(cur_w_cnt == 0)              & (cur_h_cnt == 1)  ?  sram_addr    :
                            //(cur_w_cnt == 0) & zero_skip_tmp    ? (sram_addr + 1) :
                            //(cur_w_cnt == 0) &~zero_skip_tmp    ? (sram_addr + 2) :
                            sram_addr + 1;
            sram_en_nxt   = (cur_w_cnt == 0) || (cur_h_cnt == img_height_size) ? 1'b0 :
                                               zero_skip_tmp                   ? 1'b0 : 1'b1;
            zero_skip_tmp_clr = zero_skip_tmp;
          end
        endcase // shift_ctrl      
      end
  end

  always @(posedge clk) begin: proc_sram_addr
    if(~reset_n) begin
      sram_addr <= 0;
    end begin
      sram_addr <= sram_addr_nxt;
    end
  end

  assign serial_output = sram_data_latch[0];
  assign serial_en     = shift_cnt_inc_dly0;
  assign shift_idle    = (fsm_state == S_IDEL) & (shift_cnt == 3'b000);

endmodule      
