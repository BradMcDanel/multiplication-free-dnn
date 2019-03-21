`timescale 1ns / 1ps

module demux1to9#(parameter CATCH_START_BIT = 10)
(
    clk,
    Data_in,
    sel,
    Data_out_0,
    Data_out_1,
    Data_out_2,
    Data_out_3,
    Data_out_4,
    Data_out_5,
    Data_out_6,
    Data_out_7,
    Data_out_8   
    );
    input clk;
    input Data_in;
    input [4:0] sel;
    output reg Data_out_0 = 0;
    output reg Data_out_1 = 0;
    output reg Data_out_2 = 0;
    output reg Data_out_3 = 0;
    output reg Data_out_4 = 0;
    output reg Data_out_5 = 0;
    output reg Data_out_6 = 0;
    output reg Data_out_7 = 0;
    output reg Data_out_8 = 0;

//always block with Data_in and sel in its sensitivity list
always @(clk)
    begin
        case (sel)
            CATCH_START_BIT : begin
                Data_out_0 = Data_in;
            end
            CATCH_START_BIT + 1 : begin
                Data_out_1 = Data_in;
            end
            CATCH_START_BIT + 2 : begin
                Data_out_2 = Data_in;
            end
            CATCH_START_BIT + 3 : begin
                Data_out_3 = Data_in;
            end
            CATCH_START_BIT + 4 : begin
                Data_out_4 = Data_in;
            end
            CATCH_START_BIT + 5 : begin
                Data_out_5 = Data_in;
            end
            CATCH_START_BIT + 6 : begin
                Data_out_6 = Data_in;
            end
            CATCH_START_BIT + 7 : begin
                Data_out_7 = Data_in;
            end
            default : begin
                Data_out_8 = Data_in;
            end
        endcase
    end

    endmodule
