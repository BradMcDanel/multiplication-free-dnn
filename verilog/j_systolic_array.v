///////////////////////////////////////
//
//
// this is the subarray module
module j_systolic_array #(
    parameter SUBARRAY_WIDTH      = 1                          , // width of the subarray
    parameter SUBARRAY_HEIGHT     = 1                          , // height of the subarray
    parameter NUM_DATAFLOW_PER_MX = 2                          ,
    parameter W_DATA_MUX          = clog2(NUM_DATAFLOW_PER_MX)
) (
    input                                                  clk             ,
    input                                                  reset           ,
    input  [                        4*SUBARRAY_HEIGHT-1:0] accumulation_in ,
    input  [                        4*SUBARRAY_HEIGHT-1:0] clr_and_plus_one,
    input  [                        4*SUBARRAY_HEIGHT-1:0] serial_end      ,
    input  [                        4*SUBARRAY_HEIGHT-1:0] mac_en          ,
    input  [       NUM_DATAFLOW_PER_MX*SUBARRAY_WIDTH-1:0] dataflow_in     ,
    input  [SUBARRAY_WIDTH*SUBARRAY_HEIGHT*W_DATA_MUX-1:0] dataflow_select ,
    input  [                           SUBARRAY_WIDTH-1:0] update_w        ,
    input  [           SUBARRAY_WIDTH*SUBARRAY_HEIGHT-1:0] control1        ,
    input  [       NUM_DATAFLOW_PER_MX*SUBARRAY_WIDTH-1:0] zero_dataflow_in,
    input  [SUBARRAY_WIDTH-1:0] zero_sel_in,
    output [                        4*SUBARRAY_HEIGHT-1:0] result          ,
    output [                        4*SUBARRAY_HEIGHT-1:0] result_en       ,
    output [                        4*SUBARRAY_HEIGHT-1:0] result_start    ,
    output [                        4*SUBARRAY_HEIGHT-1:0] result_end
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

// create the systolic array
wire [NUM_DATAFLOW_PER_MX-1:0] arr_dataflow_in       [SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
wire [NUM_DATAFLOW_PER_MX-1:0] arr_dataflow_out      [SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
wire [                    3:0] arr_clr_and_plus_one_i[SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
wire [                    3:0] arr_clr_and_plus_one_o[SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
wire [                    3:0] arr_serial_ende_i     [SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
wire [                    3:0] arr_serial_ende_o     [SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
wire                           arr_update_w_i        [SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
wire                           arr_update_w_o        [SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
wire [                    3:0] arr_mac_en_i          [SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
wire [                    3:0] arr_mac_en_o          [SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
wire [                    3:0] arr_accumulation_in   [SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
wire [                    3:0] arr_result            [SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];

genvar i,j;
generate
    for (j=0; j<SUBARRAY_HEIGHT; j=j+1) begin: g_H
        for (i=0; i<SUBARRAY_WIDTH; i=i+1) begin: g_W

            assign arr_update_w_i[i][j] = update_w[i];

        if(j==0) begin: g_j_eq_0
            assign arr_dataflow_in       [i][0] = dataflow_in      [i*NUM_DATAFLOW_PER_MX +: NUM_DATAFLOW_PER_MX];
            //assign arr_update_w_i        [i][0] = update_w         [i];
        end else begin: g_j_others
            assign arr_dataflow_in       [i][j] = arr_dataflow_out       [i][j-1];
            //assign arr_update_w_i        [i][j] = arr_update_w_o         [i][j-1];
        end

        if(i==0) begin: g_i_eq_0
            assign arr_accumulation_in   [0][j] = accumulation_in  [4*j +: 4];
            assign arr_mac_en_i          [0][j] = mac_en           [4*j +: 4];
            assign arr_clr_and_plus_one_i[0][j] = clr_and_plus_one [4*j +: 4];
        end else begin: g_i_others
            assign arr_accumulation_in   [i][j] = arr_result             [i-1][j];
            assign arr_mac_en_i          [i][j] = arr_mac_en_o           [i-1][j];
            assign arr_clr_and_plus_one_i[i][j] = arr_clr_and_plus_one_o [i-1][j];
        end

        if(i==(SUBARRAY_WIDTH-1)) begin: g_i_eq_W
            assign result                [4*j +: 4] = arr_result             [SUBARRAY_WIDTH-1][j];
            assign result_en             [4*j +: 4] = arr_mac_en_o           [SUBARRAY_WIDTH-1][j];
            assign result_start          [4*j +: 4] = arr_clr_and_plus_one_o [SUBARRAY_WIDTH-1][j];
        end

        j_MX_cell #(.DATA_WIDTH(NUM_DATAFLOW_PER_MX)) i_j_MX_cell (
            .clk               (clk                                                             ),
            .reset             (reset                                                           ),
            .dataflow_select   (dataflow_select[((SUBARRAY_WIDTH*j)+i)*W_DATA_MUX +: W_DATA_MUX]),
            .control1          (control1       [((SUBARRAY_WIDTH*j)+i)]                         ),
            .dataflow_in       (arr_dataflow_in       [i][j]                                    ),
            .dataflow_out      (arr_dataflow_out      [i][j]                                    ),
            .clr_and_plus_one_i(arr_clr_and_plus_one_i[i][j]                                    ),
            .clr_and_plus_one_o(arr_clr_and_plus_one_o[i][j]                                    ),
            .update_w_i        (arr_update_w_i        [i][j]                                    ),
            .update_w_o        (arr_update_w_o        [i][j]                                    ),
            .mac_en_i          (arr_mac_en_i          [i][j]                                    ),
            .mac_en_o          (arr_mac_en_o          [i][j]                                    ),
            .accumulation_in   (arr_accumulation_in   [i][j]                                    ),
            .result            (arr_result            [i][j]                                    )
        );

        end
    end
endgenerate
endmodule




