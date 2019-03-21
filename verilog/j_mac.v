//`timescale 1ns / 1ps


module j_mac #(
  parameter WEIGHT_RESET_VAL = 8'b00100011,
  parameter SHARED_W         = 0
) (
  input             clk               ,
  input             reset             ,
  input             accumulation      ,
  input             plus_one          ,
  input             clear_accu_control,
  input             dataflow_in       , //it's the input
  input             control1          , // make the control to 1'b1;
  input             update_w          ,
  input             mac_en            ,
  input       [7:0] shared_W          ,
  output reg        result            ,
  output wire       input_accu_adder
);

    //output wire dd2;
    wire       dd2        ;
    wire       dd1,dd3,dd4;
    wire [7:0] Sum        ;
    wire [7:0] B          ;
    wire [7:0] transferC  ;
    wire       Cout,cc_in;
    reg  [7:0] W          ;
    reg        p0         ;
    reg        p1         ;
    reg        p2         ;
    reg        p3         ;
    reg        p4         ;
    reg        p5         ;
    reg        p6         ;
    reg        p7         ;
    reg        f_in       ;
    reg        q2         ;
    reg        Sum0_reg   ;
    reg        result_d   ;
    reg        Cin_adder  ;
    reg        Cin_adder_d;
    wire       z0,z1,z2,z3,z4,z5,z6,z7;
    wire       Sum0_dummy ;

    generate if(SHARED_W == 0) begin: g_W
        always @(posedge clk) begin : proc_W
          if(reset) begin
            W <= WEIGHT_RESET_VAL;
          end else if(update_w) begin
            W <= {dataflow_in, W[7:1]};
          end
        end
      end
      else begin
        always @(*) begin : proc_W
          W = shared_W;
        end
      end 
    endgenerate

    assign cc_in      = f_in;
    assign Sum0_dummy = Sum0_reg;
    assign dd1        = ~dd2;
    assign dd2        = Sum[0];
    //assign dd3        = q2;
    assign dd3        = dd2;
    //assign dd4        = result_d;
    wire Cin_adder_d_mask = Cin_adder_d & ~clear_accu_control;
    assign dd4 = mac_en ? (dd1 ^ plus_one ^ Cin_adder_d_mask) : 1'b0;

    //assign cc_in = f_in;

    assign z0 = W[0] & dataflow_in /*cc_in*/;
    assign z1 = W[1] & dataflow_in /*cc_in*/;
    assign z2 = W[2] & dataflow_in /*cc_in*/;
    assign z3 = W[3] & dataflow_in /*cc_in*/;
    assign z4 = W[4] & dataflow_in /*cc_in*/;
    assign z5 = W[5] & dataflow_in /*cc_in*/;
    assign z6 = W[6] & dataflow_in /*cc_in*/;
    assign z7 = W[7] & dataflow_in /*cc_in*/;

    assign B[0] = p0 & ~clear_accu_control;
    assign B[1] = p1 & ~clear_accu_control;
    assign B[2] = p2 & ~clear_accu_control;
    assign B[3] = p3 & ~clear_accu_control;
    assign B[4] = p4 & ~clear_accu_control;
    assign B[5] = p5 & ~clear_accu_control;
    assign B[6] = p6 & ~clear_accu_control;
    assign B[7] = p7 & ~clear_accu_control;
    
    // control the input to the accumulation adder    
    assign input_accu_adder = (~control1&dd4)|(control1&dd3);

    fullAdder FA1 (
      .In1 (z0          ),
      .In2 (B[0]        ),
      .Cin (1'b0        ),
      .Sum (Sum[0]      ),
      .Cout(transferC[0])
    );
    fullAdder FA2 (
      .In1 (z1          ),
      .In2 (B[1]        ),
      .Cin (transferC[0]),
      .Sum (Sum[1]      ),
      .Cout(transferC[1])
    );
    fullAdder FA3 (
      .In1 (z2          ),
      .In2 (B[2]        ),
      .Cin (transferC[1]),
      .Sum (Sum[2]      ),
      .Cout(transferC[2])
    );
    fullAdder FA4 (
      .In1 (z3          ),
      .In2 (B[3]        ),
      .Cin (transferC[2]),
      .Sum (Sum[3]      ),
      .Cout(transferC[3])
    );
    fullAdder FA5 (
      .In1 (z4          ),
      .In2 (B[4]        ),
      .Cin (transferC[3]),
      .Sum (Sum[4]      ),
      .Cout(transferC[4])
    );
    fullAdder FA6 (
      .In1 (z5          ),
      .In2 (B[5]        ),
      .Cin (transferC[4]),
      .Sum (Sum[5]      ),
      .Cout(transferC[5])
    );
    fullAdder FA7 (
      .In1 (z6          ),
      .In2 (B[6]        ),
      .Cin (transferC[5]),
      .Sum (Sum[6]      ),
      .Cout(transferC[6])
    );
    fullAdder FA8 (
      .In1 (z7          ),
      .In2 (B[7]        ),
      .Cin (transferC[6]),
      .Sum (Sum[7]      ),
      .Cout(Cout        )
    );
    always@(posedge clk)
      if(reset) begin
        p0   <= 0;
        p1   <= 0;
        p2   <= 0;
        p3   <= 0;
        p4   <= 0;
        p5   <= 0;
        p6   <= 0;
        p7   <= 0;
        f_in <= 0;
      end
      else if(mac_en) begin
        p0   <= Sum[1];
        p1   <= Sum[2];
        p2   <= Sum[3];
        p3   <= Sum[4];
        p4   <= Sum[5];
        p5   <= Sum[6];
        p6   <= Sum[7];
        p7   <= Cout;
        f_in <= dataflow_in;
      end

    always@(posedge clk) begin
      if(reset) begin
        q2       <= 0;
        Sum0_reg <= 0;
      end
      else if(mac_en) begin
        q2       <= dd2;
        Sum0_reg <= Sum[0];
      end
    end

    // this is the MAC adder
    wire Cin_adder_mask = Cin_adder & ~clear_accu_control;
    wire result_nxt     = input_accu_adder ^ accumulation ^ (Cin_adder_mask);  //SUM;
    wire Cin_adder_nxt  = (input_accu_adder & Cin_adder_mask) | (accumulation & Cin_adder_mask) | (input_accu_adder & accumulation);

    always@(posedge clk)
      if(reset) begin
        Cin_adder <= 0;
      end
      else if(mac_en) begin
        Cin_adder <= Cin_adder_nxt;  //CARRY
      end


    always@(posedge clk)
      if(reset) begin
        result    <= 0;
      end
      else if(mac_en) begin
        result    <= result_nxt;
      end

    // this is the bitserial adder for the 2's complement
    wire Cin_adder_d_nxt = (dd1 & Cin_adder_d_mask) | (plus_one & Cin_adder_d_mask) | (dd1 & plus_one);

    always@(posedge clk)
      if(reset) begin
        result_d    <= 0;
        Cin_adder_d <= 0;
      end
      else if (mac_en) begin
        result_d    <=  dd1 ^ plus_one ^ Cin_adder_d_mask;  //SUM
        Cin_adder_d <= Cin_adder_d_nxt;  //CARRY
      end

      reg  [5:0] timer;
      wire [5:0] timer_nxt;
      assign     timer_nxt = mac_en             ? timer + 1 :
                             0;
      always @(posedge clk) begin : proc_timer
        if(reset) begin
          timer <= 0;
        end else begin
          timer <= timer_nxt;
        end
      end

  endmodule







