module j_max_power_wrapper #(
	parameter SUBARRAY_WIDTH      = 32                          , // width of the subarray
	parameter SUBARRAY_HEIGHT     = 32                          , // height of the subarray
	parameter NUM_DATAFLOW_PER_MX = 8                           ,
	parameter W_DATA_MUX          = clog2(NUM_DATAFLOW_PER_MX)
) (
	input      clk          ,
	input      reset        ,
	output reg result_xor   ,
	output reg result_en_xor
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

	reg  [                        4*SUBARRAY_HEIGHT-1:0] accumulation_in                                                       ;
	reg  [                        4*SUBARRAY_HEIGHT-1:0] clr_and_plus_one                                                      ;
	reg  [                        4*SUBARRAY_HEIGHT-1:0] mac_en           = {(4*SUBARRAY_HEIGHT){1'b1}}                        ;
	reg  [       NUM_DATAFLOW_PER_MX*SUBARRAY_WIDTH-1:0] dataflow_in                                                           ;
	reg  [SUBARRAY_WIDTH*SUBARRAY_HEIGHT*W_DATA_MUX-1:0] dataflow_select  = {(SUBARRAY_WIDTH*SUBARRAY_HEIGHT*W_DATA_MUX){1'b0}};
	reg  [                           SUBARRAY_WIDTH-1:0] update_w         = {(SUBARRAY_WIDTH){1'b0}}                           ;
	reg  [           SUBARRAY_WIDTH*SUBARRAY_HEIGHT-1:0] control1         = {(SUBARRAY_WIDTH){1'b0}}                           ;
	wire [                        4*SUBARRAY_HEIGHT-1:0] result                                                                ;
	wire [                        4*SUBARRAY_HEIGHT-1:0] result_en                                                             ;

	always @(posedge clk) begin : proc_output
		if(reset) begin
			result_xor    <= 1'b0;
			result_en_xor <= 1'b0;
		end else begin
			result_xor    <= ^result;
			result_en_xor <= ^result_en;
		end
	end

	generate 
	always @(posedge clk) begin : proc_
		if(reset) begin
			 accumulation_in  <= {(4*SUBARRAY_HEIGHT){1'b0}};
			 clr_and_plus_one <= {{(4*SUBARRAY_HEIGHT-1){1'b0}}, 1'b1};
			 dataflow_in      <= {(NUM_DATAFLOW_PER_MX*SUBARRAY_WIDTH){1'b0}};
		end else begin
			accumulation_in   <= ~accumulation_in;
			clr_and_plus_one  <= {clr_and_plus_one[4*SUBARRAY_HEIGHT-2:0], clr_and_plus_one[4*SUBARRAY_HEIGHT-1]};
			dataflow_in       <= ~dataflow_in;
		end
	end
	endgenerate

	j_systolic_array #(
		.SUBARRAY_WIDTH     (SUBARRAY_WIDTH     ),
		.SUBARRAY_HEIGHT    (SUBARRAY_HEIGHT    ),
		.NUM_DATAFLOW_PER_MX(NUM_DATAFLOW_PER_MX)
	) i_j_systolic_array (
		.clk             (clk             ),
		.reset           (reset           ),
		.accumulation_in (accumulation_in ),
		.clr_and_plus_one(clr_and_plus_one),
		.mac_en          (mac_en          ),
		.dataflow_in     (dataflow_in     ),
		.dataflow_select (dataflow_select ),
		.update_w        (update_w        ),
		.control1        (control1        ),
		.result          (result          ),
		.result_en       (result_en       )
	);

endmodule